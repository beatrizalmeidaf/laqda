import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score,f1_score,recall_score, accuracy_score, roc_auc_score

def euclidean_dist(x, y):
    """Calcula a distância euclidiana ao quadrado entre dois tensores.
    Args:
        x: Tensor de forma (N, D)
        y: Tensor de forma (M, D)
    Returns:
        Tensor de forma (N, M) com as distâncias euclidianas ao quadrado entre cada par de pontos.
    """
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

class Loss_fn(torch.nn.Module):
    """
    Classe que define a função de perda customizada e o cálculo das métricas
    para o modelo de few-shot learning.
    """
    def __init__(self, args):
        super(Loss_fn, self).__init__()
        # armazena os hiperparâmetros do modelo para uso no forward pass
        self.args = args

    def forward(self, model_outputs, labels):
        """
        Define a forward pass da função de perda.
        
        Args:
            model_outputs (tuple): A tupla retornada pelo forward do MyModel.
            labels (list/tensor): Os rótulos verdadeiros para o episódio.
        """
        # desempacotar as saídas do modelo 
        support_size = self.args.numNWay * self.args.numKShot
        # desempacota a tupla vinda do modelo, cada variável corresponde a uma saída do MyModel
        prototypes, q_re, topk_acc, original_prototypes, sampled_data = model_outputs

        # cálculo da Perda Principal (Classificação do Query Set) 
        # calcula a distância de cada embedding de consulta ('q_re') para cada protótipo refinado
        dists = euclidean_dist(q_re, prototypes)
        # converte as distâncias em log-probabilidades, menor distância -> maior probabilidade
        log_p_y = F.log_softmax(-dists, dim=1)  # Shape: (num_query, num_class)
    
        # pega os rótulos verdadeiros apenas para o conjunto de consulta
        query_labels = labels[support_size:]
        device = model_outputs[0].device 
        # converte os rótulos de consulta para o formato one-hot
        query_labels = torch.tensor(query_labels, dtype=torch.float).to(device)
        
        # calcula a perda de entropia cruzada manualmente
        loss = - query_labels * log_p_y
        loss = loss.mean() 

        # cálculo da Perda Auxiliar (Regularização do Sampler) 
        # o objetivo é garantir que os dados amostrados ('sampled_data') sejam classificáveis em relação aos protótipos ORIGINAIS (antes do refinamento)
        generate_labels = torch.tensor(range(self.args.numNWay)).unsqueeze(dim=1)
        generate_labels = generate_labels.repeat(1, sampled_data.shape[1]).view(-1)
        generate_labels = F.one_hot(generate_labels, self.args.numNWay).float().to(device)
        
        # reorganiza os dados amostrados para o cálculo de distância
        sampled_data = sampled_data.view(-1, sampled_data.shape[2])
        # calcula a distância dos dados amostrados para os protótipos originais
        g_dists = euclidean_dist(sampled_data, original_prototypes)
        glog_p_y = F.log_softmax(-g_dists, dim=1)
        # calcula a perda de entropia cruzada para os dados amostrados
        g_loss = - generate_labels * glog_p_y
        g_loss = g_loss.mean()

        # combinação das Perdas
        # a perda final é a soma da perda principal com a perda auxiliar (com um peso menor)
        overall_loss = loss + 0.1 * g_loss

        # cálculo das Métricas de Avaliação 
        # converte as log-probabilidades em uma predição "hard" (one-hot)
        # encontra a log-probabilidade máxima para cada amostra
        x, _ = torch.max(log_p_y, dim=1, keepdim=True)
        one = torch.ones_like(log_p_y)
        zero = torch.zeros_like(log_p_y)
        y_pred = torch.where(log_p_y >= x, one, log_p_y)
        y_pred = torch.where(y_pred < x, zero, y_pred)

        target_mode = 'macro' # define a média para métricas multi-classe

        query_labels = query_labels.cpu().detach()
        y_pred = y_pred.cpu().detach()
        
        # calcula as métricas usando as funções do scikit-learn
        p = precision_score(query_labels, y_pred, average=target_mode, zero_division=0)
        r = recall_score(query_labels, y_pred, average=target_mode, zero_division=0)
        f = f1_score(query_labels, y_pred, average=target_mode, zero_division=0)
        acc = accuracy_score(query_labels, y_pred)

        y_score = F.softmax(-dists, dim=1) 
        y_score = y_score.cpu().detach()
        auc = roc_auc_score(query_labels, y_score)

        return overall_loss, p, r, f, acc, auc, topk_acc