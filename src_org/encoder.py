import copy
import math
import torch
import argparse
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from transformers import BertTokenizer, BertConfig, BertModel

def cosine_similarity_dist(x, y):
    """
    Calcula a similaridade de cosseno entre dois tensores.
    Args:
        x: Tensor de forma (N, D)
        y: Tensor de forma (M, D)
    Returns:
        Tensor de forma (N, M) com as similaridades de cosseno.
    """
    # Normaliza os vetores para ter norma L2 igual a 1
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    
    # Calcula o produto escalar (dot product) entre todos os pares
    # (N, D) @ (D, M) -> (N, M)
    similarity_matrix = torch.mm(x_norm, y_norm.transpose(0, 1))
    
    return similarity_matrix

def euclidean_dist(x, y):
    '''
    Calcula a distância euclidiana ao quadrado entre dois tensores.
    '''
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception("As dimensões dos vetores não batem!")

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

class BertEncoder(nn.Module):
    """
    Classe que encapsula o modelo BERT para atuar como um codificador de texto.
    """
    def __init__(self, args):
        super(BertEncoder, self).__init__()

        do_lower_case = True if args.fileVocab and "uncased" in args.fileVocab else False
        self.tokenizer = BertTokenizer.from_pretrained(args.fileVocab, do_lower_case=do_lower_case) 
        
        config = BertConfig.from_pretrained(args.fileModelConfig)
        self.bert = BertModel.from_pretrained(args.fileModel, config=config)
        
        self.hidden_size = config.hidden_size 
        
        self.lin = nn.Linear(self.hidden_size, self.hidden_size)
        
        # copia a última camada de atenção
        last_layer_index = config.num_hidden_layers - 1
        self.attention = copy.deepcopy(self.bert.encoder.layer[last_layer_index])
        
        self.drop = nn.Dropout(config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1) 
        
        self.la = args.la
        
        if args.numFreeze > 0:
            num_layers_to_freeze = min(args.numFreeze, config.num_hidden_layers) 
            if num_layers_to_freeze != args.numFreeze:
                 print(f"Aviso: Reduzindo numFreeze de {args.numFreeze} para {num_layers_to_freeze} (máximo para este modelo)")
            self.freeze_layers(num_layers_to_freeze, config.num_hidden_layers)

    def freeze_layers(self, numFreeze, total_layers):
        """ Congela as primeiras 'numFreeze' camadas do BERT. """
        unfreeze_layers = ["pooler"]
        for i in range(numFreeze, total_layers): 
            unfreeze_layers.append("layer." + str(i))

        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break 

    def forward(self, text, task_classes):
        """ Define o forward pass do encoder. """
        sentence = [x for x in text]
        
        tokenizer_output = self.tokenizer(
            sentence, padding=True, truncation=True, max_length=250, return_tensors='pt'
        )
        
        input_ids = tokenizer_output['input_ids'].to(self.bert.device)
        token_type_ids = tokenizer_output['token_type_ids'].to(self.bert.device)
        attention_mask = tokenizer_output['attention_mask'].to(self.bert.device)
        
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        last_hidden_state = outputs.last_hidden_state # Shape: (batch_size, seq_len, hidden_size)
        
        # Se LA estiver desativado, retorna o embedding do [CLS]
        if self.la == 0:
            return last_hidden_state[:, 0, :] 

        # lógica 'label-aware' ATIVADA (la=1) 
        
        task_labels_inputs = self.tokenizer(
            task_classes, return_tensors='pt', padding=True, truncation=True
        )
        task_labels_inputs_input_ids = task_labels_inputs['input_ids'].to(self.bert.device)
        task_labels_inputs_token_type_ids = task_labels_inputs['token_type_ids'].to(self.bert.device)
        task_labels_inputs_attention_mask = task_labels_inputs['attention_mask'].to(self.bert.device)

        label_outputs = self.bert(
            task_labels_inputs_input_ids,
            task_labels_inputs_attention_mask,
            task_labels_inputs_token_type_ids
        )
        # Usa o CLS token do label
        label_embedding = label_outputs.last_hidden_state[:, 0, :] # Shape: (num_classes, hidden_size)

        # Recriando a lógica original de concatenação e atenção com hidden_size dinâmico
        sentence_embeddings = last_hidden_state
        # Expande os embeddings dos labels para cada sentença no batch
        label_embeddings_expanded = label_embedding.unsqueeze(0).expand(sentence_embeddings.shape[0], -1, -1) 
        
        # Concatena ao longo da dimensão da sequência
        connect_embeddings = torch.cat((sentence_embeddings, label_embeddings_expanded), dim=1)
        # Shape: (batch_size, seq_len + num_classes, hidden_size)

        # Cria a máscara de atenção para a sequência concatenada
        sentence_mask = attention_mask 
        label_mask = torch.ones(label_embeddings_expanded.shape[0], label_embeddings_expanded.shape[1], device=self.bert.device)
        connect_mask = torch.cat((sentence_mask, label_mask), dim=1)
        # Formato para camada de atenção do BERT: (batch, 1, 1, seq_len_total)
        extended_attention_mask = connect_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Passa pela camada linear 
        linear_output = self.lin(connect_embeddings) # Espera (..., hidden_size) -> (..., hidden_size)
        dropped_output = self.drop(linear_output)
        
        # Passa pela camada de atenção copiada
        attention_output = self.attention(dropped_output, extended_attention_mask)[0] 
        
        # Conexão residual 
        residual_output = 0.1 * attention_output + 0.9 * connect_embeddings
        
        # Pega o embedding do [CLS] token (primeiro token da sequência original)
        cls_embedding_final = residual_output[:, 0, :] 
        
        return cls_embedding_final

class Sampler(nn.Module):
    """
    Módulo da rede neural que implementa a lógica de amostragem transdutiva (QDA).
    """
    def __init__(self, args, hidden_size=768): 
        super(Sampler, self).__init__()
        self.nway = args.numNWay
        self.kshot = args.numKShot
        self.qshot = args.numQShot
        self.dim = hidden_size 
        self.k = args.k
        self.num_sampled = args.sample 

    def calculate_var(self, features):
        """ Calcula a média e a variância de um conjunto de features. """
        v_mean = features.mean(dim=1) 
        v_cov = []
        for i in range(features.shape[0]):
            diag = torch.var(features[i], dim=0)
            v_cov.append(diag)
        v_cov = torch.stack(v_cov) 
        return v_mean, v_cov
        
    def forward(self, support_embddings, query_embeddings):
        # Calcular similaridade de cosseno
        similarity = cosine_similarity_dist(support_embddings, query_embeddings) # Shape (NK, NQ)
        
        # Reorganiza o tensor de similaridade
        similarity = similarity.view(self.nway, self.kshot, -1) # Shape: (N, K, NQ)
    
        # Encontrar os 'k' vizinhos MAIS SIMILARES (largest=True)
        values, indices = similarity.topk(self.k, dim=2, largest=True, sorted=True)  
        
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
        
        acc = torch.tensor(acc).mean() if acc else torch.tensor(0.0) # Evita erro se lista vazia
        
        nindices = indices.view(-1, self.k) 
        convex_feat = []
        # garante que query_embeddings tenha gradientes se necessário
        # query_embeddings_detached = query_embeddings.detach() # Descomentar se causar problemas de gradiente
        for i in range(nindices.shape[0]):
            try:
                 # convex_feat.append(query_embeddings_detached.index_select(0, nindices[i])) 
                 convex_feat.append(query_embeddings.index_select(0, nindices[i])) 
            except IndexError as e:
                 print(f"Erro no index_select: i={i}, nindices[i]={nindices[i]}, query_embeddings.shape={query_embeddings.shape}")
                 raise e
        convex_feat = torch.stack(convex_feat) 
        
        sampled_data = convex_feat.view(self.nway, self.kshot * self.k, self.dim) 
        
        return sampled_data, acc

    def distribution_calibration(self, query, base_mean, base_cov, alpha=0.21):
        calibrated_mean = (query + base_mean) / 2
        calibrated_cov = base_cov
        return calibrated_mean, calibrated_cov


class MyModel(nn.Module):
    """
    A classe principal do modelo, que une o BertEncoder e o Sampler.
    (CORRIGIDO para passar hidden_size para o Sampler)
    """
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.args = args
        self.bert = BertEncoder(args)
        self.sampler = Sampler(args, hidden_size=self.bert.hidden_size) 

    def forward(self, text, label):
        support_size = self.args.numNWay * self.args.numKShot

        text_embedding = self.bert(text, label) # Passa 'label' que contém as strings das classes do episódio
        
        support_embddings = text_embedding[:support_size] 
        query_embeddings = text_embedding[support_size:]   
        
        # Calcula protótipos originais
        # Precisamos garantir que numKShot seja pelo menos 1
        k_shot_dim = max(1, self.args.numKShot) # Evita erro se kshot for 0 
        c_prototypes = support_embddings.view(self.args.numNWay, k_shot_dim, -1) 
        original_prototypes = c_prototypes.mean(dim=1) 

        # Usa o Sampler (QDA)
        sampled_data, acc = self.sampler(support_embddings, query_embeddings)

        # Cria protótipos refinados (Support + QDA Samples)
        prototypes_data = torch.cat((c_prototypes, sampled_data), dim=1) 
        prototypes = torch.mean(prototypes_data, dim=1) 

        return (prototypes, query_embeddings, acc, original_prototypes, sampled_data)

    def visual(self, text, label): # Adicionado 'label' se la=1 for usado
        """ Método auxiliar para visualização """
        support_size = self.args.numNWay * self.args.numKShot
        
        text_embedding = self.bert(text, label) 
    
        support_embddings = text_embedding[:support_size]
        query_embeddings = text_embedding[support_size:]

        sampled_data, acc = self.sampler(support_embddings, query_embeddings)
        # sampled_data = sampled_data.to(self.bert.device) 

        k_shot_dim = max(1, self.args.numKShot)
        c_prototypes = support_embddings.view(self.args.numNWay, k_shot_dim, -1)
        
        prototypes_data = torch.cat((c_prototypes, sampled_data), dim=1)
        prototypes = torch.mean(prototypes_data, dim=1)
        
        return support_embddings, query_embeddings, sampled_data, prototypes