import csv
import os
import torch
import json
import random
import numpy as np

from tqdm import tqdm
from utils import get_parser
from losses import Loss_fn
from encoder import MyModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from data_loader import MyDataset, KShotTaskSampler, get_label_dict
from collections import defaultdict
# from tensorboardX import SummaryWriter


def set_seed(seed):
    """
    Define as seeds para todas as bibliotecas de aleatoriedade
    para garantir que os resultados do experimento sejam reprodutíveis.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def init_dataloader(args, mode):
    """
    Inicializa e retorna um DataLoader para um modo específico (train, valid, test).
    """
    # constrói o caminho completo para o arquivo de dados (.json)
    filePath = os.path.join(args.dataFile, mode + '.json')
    # define quantos episódios terá por época, dependendo do modo
    if mode == 'train' or mode == 'valid':
        episode_per_epoch = args.episodeTrain
    else:
        episode_per_epoch = args.episodeTest
    
    # cria a instância do Dataset customizado
    dataset = MyDataset(filePath)
    # cria a instância do nosso Sampler customizado que gera os episódios few-shot
    sampler = KShotTaskSampler(dataset, episodes_per_epoch=episode_per_epoch, n=args.numKShot, k=args.numNWay, q=args.numQShot, num_tasks=1)

    return sampler


def save_list_to_file(path, thelist):
    """
    Função auxiliar para salvar uma lista em um arquivo de texto, um item por linha.
    """
    # abre o arquivo em modo 'append' (a+), que adiciona ao final do arquivo sem apagar o conteúdo
    with open(path, 'a+') as f:
        for item in thelist:
            f.write("%s\n" % item)


def init_model(args):
    """
    Inicializa o modelo e o move para GPU ou CPU conforme disponível.
    """
  
    if args.numDevice >= 0 and torch.cuda.is_available():
        print(f"Usando GPU: {args.numDevice}")
        device = torch.device('cuda', args.numDevice)
        torch.cuda.set_device(device)
    else:
        print("Usando CPU")
        device = torch.device('cpu')

    # cria uma instância do modelo (MyModel) e o envia para o dispositivo
    model = MyModel(args).to(device)
    return model

def init_optim(args, model):
    """
    Inicializa o otimizador AdamW.
    """
    # define os nomes dos parâmetros que não terão decaimento de peso (weight decay)
    no_decay = ['bias', 'LayerNorm.weight']
    # separa os parâmetros do modelo em dois grupos: os que terão decaimento de peso e os que não terão
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # cria a instância do otimizador AdamW
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    return optimizer

def init_lr_scheduler(args, optim):
    """
    Inicializa o agendador de taxa de aprendizado (learning rate scheduler).
    """
    # calcula o número total de passos de treinamento
    t_total = args.epochs * args.episodeTrain
    # cria um agendador que aumenta linearmente a taxa de aprendizado por `warmup_steps` e depois a diminui linearmente até o final do treinamento
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    return scheduler


def deal_data(support_set, query_set, episode_labels, labels_dict):
    """
    Processa os dados de um lote (episódio) para o formato que o modelo espera.
    Principalmente, converte os rótulos de texto para o formato one-hot.
    """
    text, labels = [], []
    # junta o texto e os rótulos do conjunto de suporte
    for x in support_set:
        text.append(x["text"])
        labels.append(x["label"])
    # junta o texto e os rótulos do conjunto de consulta
    for x in query_set:
        text.append(x["text"])
        labels.append(x["label"])  
    
    label_ids = []
    for label in labels:
        tmp = []
        for l in episode_labels:
            if l == labels_dict[label]:
                tmp.append(1)
            else:
                tmp.append(0)
        label_ids.append(tmp)

    return text, label_ids

def train(args, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    """
    A função principal que executa o loop de treinamento e validação.
    """
    if val_dataloader is None:
        acc_best_state = None
    
    train_loss, train_acc, train_p, train_r, train_f1, train_auc, train_topkacc = [], [], [], [], [], [], []
    epoch_train_loss, epoch_train_acc, epoch_train_p, epoch_train_r, epoch_train_f1, epoch_train_auc, epoch_train_topkacc = [], [], [], [], [], [], []
    val_loss, val_acc, val_p, val_r, val_f1, val_auc, val_topkacc = [], [], [], [], [], [], []
    epoch_val_loss, epoch_val_acc, epoch_val_p, epoch_val_r, epoch_val_f1, epoch_val_auc, epoch_val_topkacc = [], [], [], [], [], [], []
    
    best_p, best_r, best_f1, best_acc, best_auc = 0, 0, 0, 0, 0
    loss_fn = Loss_fn(args)
    
    acc_best_model_path = os.path.join(args.fileModelSave, 'acc_best_model.pth')
    cycle = 0
    labels_dict = get_label_dict(args)
    id2label = {y: x for x, y in labels_dict.items()}

    for epoch in range(args.epochs):
        print('=== Época: {} ==='.format(epoch))
        model.train() 
        
        if cycle == args.patience:
            break

        for i, batch in tqdm(enumerate(tr_dataloader)):
            optim.zero_grad()
            support_set, query_set, episode_labels = batch
            
            # converte os rótulos para string para o tokenizador
            label_text = [str(id2label[int(el)]) for el in episode_labels]
            
            text, labels = deal_data(support_set, query_set, episode_labels, labels_dict)
            
            model_outputs = model(text, label_text)
            loss, p, r, f, acc, auc, topk_acc = loss_fn(model_outputs, labels)
            
            loss.backward()
            optim.step()
            lr_scheduler.step()
            
            train_loss.append(loss.item())
            train_p.append(p)
            train_r.append(r)
            train_f1.append(f)
            train_acc.append(acc)
            train_auc.append(auc)
            train_topkacc.append(topk_acc)

            print('Train Loss: {}, Train p: {}, Train r: {}, Train f1: {},  Train acc: {},  Train auc: {}, Train topk acc: {}'.format(loss, p, r, f, acc, auc, topk_acc))

        avg_loss = np.mean(train_loss[-args.episodeTrain:])
        avg_acc = np.mean(train_acc[-args.episodeTrain:])
        avg_p = np.mean(train_p[-args.episodeTrain:])
        avg_r = np.mean(train_r[-args.episodeTrain:])
        avg_f1 = np.mean(train_f1[-args.episodeTrain:])
        avg_auc = np.mean(train_auc[-args.episodeTrain:])
        avg_topkacc = np.mean(train_topkacc[-args.episodeTrain:])
        print('Média Train Loss: {}, Média Train p: {}, Média Train r: {}, Média Train f1: {}, Média Train acc: {}, Média Train auc: {}, Média Train topk acc: {}'.format(avg_loss, avg_p, avg_r, avg_f1, avg_acc, avg_auc, avg_topkacc))
        
        epoch_train_loss.append(avg_loss)
        epoch_train_acc.append(avg_acc)
        epoch_train_p.append(avg_p)
        epoch_train_r.append(avg_r)
        epoch_train_f1.append(avg_f1)
        epoch_train_auc.append(avg_auc)
        epoch_train_topkacc.append(avg_topkacc)

        if val_dataloader is None:
            continue 

        with torch.no_grad():
            model.eval()
            
            for batch in tqdm(val_dataloader):
                support_set, query_set, episode_labels = batch
                text, labels = deal_data(support_set, query_set, episode_labels, labels_dict)

                # converte os rótulos para string para o tokenizador
                label_text = [str(id2label[int(el)]) for el in episode_labels]
                
                model_outputs = model(text, label_text)
                loss, p, r, f, acc, auc, topkacc = loss_fn(model_outputs, labels)
                
                val_loss.append(loss.item())
                val_acc.append(acc)
                val_p.append(p)
                val_r.append(r)
                val_f1.append(f)
                val_auc.append(auc)
                val_topkacc.append(topkacc)
            
            avg_loss = np.mean(val_loss[-args.episodeTrain:])
            avg_acc = np.mean(val_acc[-args.episodeTrain:])
            avg_p = np.mean(val_p[-args.episodeTrain:])
            avg_r = np.mean(val_r[-args.episodeTrain:])
            avg_f1 = np.mean(val_f1[-args.episodeTrain:])
            avg_auc = np.mean(val_auc[-args.episodeTrain:])
            avg_topkacc = np.mean(val_topkacc[-args.episodeTrain:])
            
            epoch_val_loss.append(avg_loss)
            epoch_val_acc.append(avg_acc)
            epoch_val_p.append(avg_p)
            epoch_val_r.append(avg_r)
            epoch_val_f1.append(avg_f1)
            epoch_val_auc.append(avg_auc)
            epoch_val_topkacc.append(avg_topkacc)

            postfix = ' (Melhor)' if avg_p >= best_p else ' (Melhor: {})'.format(best_p)
            r_prefix = ' (Melhor)' if avg_r >= best_r else ' (Melhor: {})'.format(best_r)
            f1_prefix = ' (Melhor)' if avg_f1 >= best_f1 else ' (Melhor: {})'.format(best_f1)
            acc_prefix = ' (Melhor)' if avg_acc >= best_acc else ' (Melhor: {})'.format(best_acc)
            auc_prefix = ' (Melhor)' if avg_auc >= best_auc else ' (Melhor: {})'.format(best_auc)
            print('Média Val Loss: {}, Média Val p: {}{}, Média Val r: {}{}, Média Val f1: {}{}, Média Val acc: {}{}, Média Val auc: {}{},  Média Val topkacc: {}'.format(
                avg_loss, avg_p, postfix, avg_r, r_prefix, avg_f1, f1_prefix, avg_acc, acc_prefix, avg_auc, auc_prefix, avg_topkacc))
            
            cycle += 1
            if avg_acc >= best_acc:
                torch.save(model.state_dict(), acc_best_model_path)
                best_acc = avg_acc
                acc_best_state = model.state_dict()
                cycle = 0
                
    for name in ['epoch_train_loss', 'epoch_train_p', 'epoch_train_r', 'epoch_train_f1', 'epoch_train_acc', 'epoch_train_auc', 'epoch_train_topkacc', 'epoch_val_loss', 'epoch_val_p', 'epoch_val_r', 'epoch_val_f1', 'epoch_val_acc', 'epoch_val_auc', 'epoch_val_topkacc']:
        save_list_to_file(os.path.join(args.fileModelSave, name + '.txt'), locals()[name])

    return model

def test(args, test_dataloader, model):
    """
    Função para testar o modelo treinado no conjunto de teste.
    """
    val_p, val_r, val_loss, val_f1, val_acc, val_auc, val_topkacc = [], [], [], [], [], [], []
    loss_fn = Loss_fn(args)
    
    with torch.no_grad(): 
        model.eval() 
        
        labels_dict = get_label_dict(args)
        id2label = {y: x for x, y in labels_dict.items()}
        
        for batch in tqdm(test_dataloader):
            support_set, query_set, episode_labels = batch
            text, labels = deal_data(support_set, query_set, episode_labels, labels_dict)

            # converte os rótulos para string para o tokenizador
            label_text = [str(id2label[int(el)]) for el in episode_labels]
            
            model_outputs = model(text, label_text)
            loss, p, r, f, acc, auc, topkacc = loss_fn(model_outputs, labels)
            
            val_loss.append(loss.item())
            val_acc.append(acc)
            val_p.append(p)
            val_r.append(r)
            val_f1.append(f)
            val_auc.append(auc)
            val_topkacc.append(topkacc)
            
    avg_loss = np.mean(val_loss)
    avg_acc = np.mean(val_acc)
    avg_p = np.mean(val_p)
    avg_r = np.mean(val_r)
    avg_f1 = np.mean(val_f1)
    avg_auc = np.mean(val_auc)
    avg_topkacc = np.mean(val_topkacc)

    print('Test p: {}'.format(avg_p))
    print('Test r: {}'.format(avg_r))
    print('Test f1: {}'.format(avg_f1))
    print('Test acc: {}'.format(avg_acc))
    print('Test auc: {}'.format(avg_auc))
    print('Test topkacc: {}'.format(avg_topkacc))
    print('Test Loss: {}'.format(avg_loss))

    path = args.fileModelSave + "/test_score.json"
    with open(args.fileModelSave+'/result.csv', 'a+', newline="") as f:
        writer = csv.writer(f)
        data = ["commont", args.commont, "data", args.dataFile,"shot", args.numKShot, "acc", avg_acc]
        writer.writerow(data)
    with open(path, "a+") as fout:
        tmp = {"commont": args.commont, "data":args.dataFile,"shot": args.numKShot, "acc": avg_acc, "p": avg_p, "r": avg_r, "f1": avg_f1, "auc": avg_auc, "Loss": avg_loss}
        fout.write("%s\n" % json.dumps(tmp, ensure_ascii=False))

def write_args_to_josn(args):
    """
    Salva os argumentos/configurações do experimento em um arquivo config.json.
    """
    path = args.fileModelSave + "/config.json"
    args_dict = vars(args) # converte o objeto de argumentos para um dicionário
    json_str = json.dumps(args_dict, indent=4) 
    with open(path, 'w') as json_file: 
        json_file.write(json_str)

def main():
    """
    A função principal que executa todo o fluxo.
    """
    args = get_parser().parse_args()

    if not os.path.exists(args.fileModelSave):
        os.makedirs(args.fileModelSave)

    # salva a configuração e define a semente aleatória
    write_args_to_josn(args)
    set_seed(42)
    
    # inicializa o modelo, dataloaders, otimizador e agendador
    model = init_model(args)
    tr_dataloader = init_dataloader(args, 'train')
    val_dataloader = init_dataloader(args, 'valid')
    test_dataloader = init_dataloader(args, 'test')
    optim = init_optim(args, model)
    lr_scheduler = init_lr_scheduler(args, optim)
    
    # inicia o treinamento
    model = train(args=args,
                  tr_dataloader=tr_dataloader,
                  val_dataloader=val_dataloader,
                  model=model,
                  optim=optim,
                  lr_scheduler=lr_scheduler)
    
    # carrega o melhor modelo salvo e inicia o teste
    model.load_state_dict(torch.load(args.fileModelSave + "/acc_best_model.pth"))
    print('Testando com o melhor modelo (baseado em acurácia)...')
    test(args=args,
         test_dataloader=test_dataloader,
         model=model)

if __name__ == '__main__':
    main()