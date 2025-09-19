import copy
import math
import torch
import argparse
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from transformers import BertTokenizer, BertConfig, BertModel

def euclidean_dist(x, y):
    '''
    Calcula a distância euclidiana ao quadrado entre dois tensores.
    
    Args:
        x (torch.Tensor): Tensor de shape (N, D), onde N é o número de vetores e D a dimensão.
        y (torch.Tensor): Tensor de shape (M, D), onde M é o número de vetores e D a dimensão.

    Returns:
        torch.Tensor: Um tensor de shape (N, M) com as distâncias.
    '''
    # N = número de amostras em x, M = número de amostras em y, D = dimensão
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception("As dimensões dos vetores não batem!")

    # broadcasting: transforma x e y para que possam ser subtraídos elemento a elemento
    # x passa a ter shape (N, M, D)
    x = x.unsqueeze(1).expand(n, m, d)
    # y passa a ter shape (N, M, D)
    y = y.unsqueeze(0).expand(n, m, d)

    # calcula (x - y)^2 para cada elemento e soma ao longo da dimensão D
    return torch.pow(x - y, 2).sum(2)

class BertEncoder(nn.Module):
    """
    Classe que encapsula o modelo BERT para atuar como um codificador de texto,
    transformando sentenças em embeddings.
    """
    def __init__(self, args):
        super(BertEncoder, self).__init__()

        # carrega o tokenizador pré-treinado do BERT a partir do caminho especificado
        self.tokenizer = BertTokenizer.from_pretrained(args.fileVocab, do_lower_case=True)
        # carrega a configuração do modelo BERT (arquitetura, tamanho)
        config = BertConfig.from_pretrained(args.fileModelConfig)
        # carrega o modelo BERT pré-treinado com os pesos e a configuração
        self.bert = BertModel.from_pretrained(args.fileModel, config=config)
        
        # faz uma cópia profunda da última camada de atenção do BERT, isso permite usar o mecanismo de atenção do BERT de forma customizada
        self.attention = copy.deepcopy(self.bert.encoder.layer[11])
        # define uma camada linear com a mesma dimensão de entrada e saída do BERT (768)
        self.lin = nn.Linear(768, 768)
        # define uma camada de Dropout para regularização evitando overfitting
        self.drop = nn.Dropout(0.1)
        # o parâmetro 'la' (label-aware) ativa uma lógica especial no forward pass
        self.la = args.la
        
        # se for especificado para congelar algumas camadas do BERT:
        if args.numFreeze > 0:
            self.freeze_layers(args.numFreeze)

    def freeze_layers(self, numFreeze):
        """
        Congela os parâmetros das primeiras 'numFreeze' camadas do BERT.
        """
        # define quais partes do modelo BERT devem permanecer treináveis
        unfreeze_layers = ["pooler"]
        for i in range(numFreeze, 12): # BERT-base tem 12 camadas (0 a 11)
            unfreeze_layers.append("layer." + str(i))

        for name, param in self.bert.named_parameters():
            # desativa o cálculo de gradientes para todos os parâmetros
            param.requires_grad = False
            # verifica se o nome do parâmetro atual pertence a uma das camadas que devem ser descongeladas
            for ele in unfreeze_layers:
                if ele in name:
                    # se pertencer reativa o cálculo de gradientes para esse parâmetro
                    param.requires_grad = True
                    break 

    def forward(self, text, task_classes):
        """
        Define o forward pass do encoder.
        """
        # converte a entrada de texto em uma lista de strings
        sentence = [x for x in text]
        
        # usa o tokenizador para converter o texto em IDs numéricos
        tokenizer_output = self.tokenizer(
            sentence,
            padding=True,       # adiciona padding para igualar o comprimento das sentenças
            truncation=True,    # trunca sentenças longas
            max_length=250,
            return_tensors='pt' # retorna como tensores do PyTorch
        )
        
        # move os tensores tokenizados para o mesmo dispositivo (CPU/GPU) que o modelo BERT
        input_ids = tokenizer_output['input_ids'].to(self.bert.device)
        token_type_ids = tokenizer_output['token_type_ids'].to(self.bert.device)
        attention_mask = tokenizer_output['attention_mask'].to(self.bert.device)
        
        # passa os dados pelo modelo BERT para obter os embeddings
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # se o modo 'label-aware' estiver DESATIVADO
        if self.la == 0:
            # retorna uma representação da sentença tirando a média dos embeddings de todos os seus tokens
            return outputs.last_hidden_state.mean(dim=1)

        # lógica do modo 'babel-aware' ATIVADO 
        # tokeniza também os nomes das classes da tarefa atual
        task_labels_inputs = self.tokenizer.batch_encode_plus(
            task_classes, return_tensors='pt', padding=True)
        task_labels_inputs_input_ids = task_labels_inputs['input_ids'].to(self.bert.device)
        task_labels_inputs_token_type_ids = task_labels_inputs['token_type_ids'].to(self.bert.device)
        task_labels_inputs_attention_mask = task_labels_inputs['attention_mask'].to(self.bert.device)

        # obtém os embeddings dos nomes das classes
        label_embedding = self.bert(
            task_labels_inputs_input_ids,
            task_labels_inputs_attention_mask,
            task_labels_inputs_token_type_ids
        ).last_hidden_state.mean(dim=1)

        # combina os embeddings das sentenças com os embeddings dos rótulos
        sentence_embeddings = outputs.last_hidden_state
        label_embeddings = label_embedding.unsqueeze(dim=0).repeat_interleave(sentence_embeddings.shape[0], dim=0)
        connect_embeddings = torch.cat((sentence_embeddings, label_embeddings), dim=1)
        
        # passa essa combinação por uma camada de atenção customizada para refinar o embedding
        outputs = self.lin(connect_embeddings)
        outputs = self.drop(outputs)
        outputs = self.attention(outputs)[0]
        # usa uma conexão residual para estabilizar o treinamento
        outputs = 0.1 * outputs + 0.9 * connect_embeddings
        
        # retorna o embedding do token [CLS] (índice 0), que é uma representação agregada da sequência
        outputs = outputs[:, 0, :]
        return outputs
        
        
class Sampler(nn.Module):
    """
    Módulo da rede neural que implementa a lógica de amostragem transdutiva.
    Ele seleciona exemplos de consulta com base na proximidade com os de suporte.
    """
    def __init__(self, args):
        super(Sampler, self).__init__()
        self.nway = args.numNWay
        self.kshot = args.numKShot
        self.qshot = args.numQShot
        self.dim = 768
        # TOP R
        self.k = args.k
        # o número de amostras
        self.num_sampled = args.sample

    def calculate_var(self, features):
        """ Calcula a média e a variância de um conjunto de features. """
        # features shape: (NK, k, 768)
        v_mean = features.mean(dim=1) # Média, shape: (NK, 768)
        v_cov = []
        for i in range(features.shape[0]):
            # calcula a variância ao longo da dimensão 0 para cada grupo de features
            diag = torch.var(features[i], dim=0)
            v_cov.append(diag)
        # empilha as variâncias em um tensor
        v_cov = torch.stack(v_cov) # shape: (NK, 768)
        return v_mean, v_cov
        
    def forward(self, support_embddings, query_embeddings):
        # calcula a distância de cada suporte para cada consulta
        similarity = euclidean_dist(support_embddings, query_embeddings) # Shape (NK, NQ)
        
        # reorganiza o tensor de similaridade para separar por classe e shot
        similarity = similarity.view(self.nway, self.kshot, -1) # Shape: (N, K, NQ)
    
        # para cada amostra de suporte, encontra os 'k' vizinhos mais próximos no conjunto de consulta
        values, indices = similarity.topk(self.k, dim=2, largest=False, sorted=True)  
        
        # calcula uma métrica de acurácia interna para ver se os vizinhos pertencem à classe certa
        acc = []
        for i in range(self.nway):
            min_index = i * self.qshot
            max_index = (i + 1) * self.qshot - 1
            for j in range(self.kshot):
                count = 0.0
                for z in range(self.k):
                    if min_index <= indices[i][j][z] <= max_index:
                        count += 1
                acc.append(count / (self.k + 0.0)) 
        
        acc = torch.tensor(acc)
        acc = acc.mean()
        
        # reorganiza os índices dos vizinhos encontrados
        nindices = indices.view(-1, self.k)
        
        # seleciona os embeddings de consulta correspondentes aos índices encontrados
        convex_feat = []
        for i in range(nindices.shape[0]):
            convex_feat.append(query_embeddings.index_select(0, nindices[i]))
        convex_feat = torch.stack(convex_feat) # Shape: (NK, k, 768)
        
        # reorganiza os dados amostrados para o formato final
        sampled_data = convex_feat.view(self.nway, self.kshot * self.k, self.dim)
        
        return sampled_data, acc

    def distribution_calibration(self, query, base_mean, base_cov, alpha=0.21):
        """
        Método para uma técnica de calibração de distribuição,
        ajustando a média e covariância.
        """
        calibrated_mean = (query + base_mean) / 2
        calibrated_cov = base_cov
        return calibrated_mean, calibrated_cov


class MyModel(nn.Module):
    """
    A classe principal do modelo, que une o BertEncoder e o Sampler.
    """
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.args = args
        self.bert = BertEncoder(args)
        self.sampler = Sampler(args)
        # self.reference = nn.Linear(768, self.args.numNWay, bias=True)

    def forward(self, text, label):
        # calcula o número total de amostras de suporte e consulta
        support_size = self.args.numNWay * self.args.numKShot
        query_size = self.args.numNWay * self.args.numQShot
        
        # usa o encoder BERT para obter embeddings de todo o texto
        text_embedding = self.bert(text, label)
        
        # separa os embeddings em conjuntos de suporte e consulta
        support_embddings = text_embedding[:support_size]
        query_embeddings = text_embedding[support_size:]
        
        # calcula os protótipos originais
        # reorganiza os embeddings de suporte para agrupar por classe (N x K x dim)
        c_prototypes = support_embddings.view(self.args.numNWay, -1, support_embddings.shape[1])
        # calcula a média dos embeddings de suporte por classe para obter os protótipos
        original_prototypes = c_prototypes.mean(dim=1)

        # usa o Sampler para selecionar os dados de consulta mais relevantes
        sampled_data, acc = self.sampler(support_embddings, query_embeddings)

        # cria os protótipos refinados
        # concatena os embeddings de suporte originais com os dados amostrados da consulta
        prototypes = torch.cat((c_prototypes, sampled_data), dim=1)
        # calcula a nova média com os dados de suporte e os dados amostrados
        prototypes = torch.mean(prototypes, dim=1)

        return (prototypes, query_embeddings, acc, original_prototypes, sampled_data)

    def visual(self, text):
        """
        Um método auxiliar que realiza um forward pass similar.
        """
        support_size = self.args.numNWay * self.args.numKShot
        query_size = self.args.numNWay * self.args.numQShot
        text_embedding = self.bert(text) 
    
        support_embddings = text_embedding[:support_size]
        query_embeddings = text_embedding[support_size:]

        sampled_data, acc = self.sampler(support_embddings, query_embeddings)
        sampled_data = sampled_data.to(self.bert.device)

        c_prototypes = support_embddings.view(self.args.numNWay, -1, support_embddings.shape[1])
        
        prototypes = torch.cat((c_prototypes, sampled_data), dim=1)
        prototypes = torch.mean(prototypes, dim=1)
        
        return support_embddings, query_embeddings, sampled_data, prototypes
