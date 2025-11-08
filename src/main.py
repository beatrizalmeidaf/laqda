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


def init_dataloader(args, mode, labels_dict): # <-- MODIFICADO: recebe labels_dict
    """
    Inicializa e retorna um Sampler para um modo específico (train, valid, test).
    """
    filePath = os.path.join(args.dataFile, mode + '.json')
    if mode == 'train' or mode == 'valid':
        episode_per_epoch = args.episodeTrain
    else:
        episode_per_epoch = args.episodeTest
    
    try:
        dataset = MyDataset(filePath, labels_dict) 
        
        if len(dataset) == 0:
            print(f"AVISO: O dataset {filePath} está vazio ou não contém labels válidas do mapa. Pulando.")
            return None 

        # verifica se o dataset tem classes suficientes para o N-way
        if dataset.num_classes() < args.numNWay:
            print(f"AVISO: O dataset {filePath} tem apenas {dataset.num_classes()} classes (das {len(labels_dict)} esperadas), mas a tarefa é {args.numNWay}-way.")
            if args.numNWay > dataset.num_classes():
                raise ValueError(f"Número insuficiente de classes ({dataset.num_classes()}) em {filePath} para a tarefa {args.numNWay}-way.")
                
        sampler = KShotTaskSampler(dataset, episodes_per_epoch=episode_per_epoch, n=args.numKShot, k=args.numNWay, q=args.numQShot, num_tasks=1)
        return sampler
    except Exception as e:
        print(f"Erro ao inicializar dataloader para {filePath}: {e}")
        raise


def save_list_to_file(path, thelist):
    """
    Função auxiliar para salvar uma lista em um arquivo de texto, um item por linha.
    """
    try:
        with open(path, 'a+') as f:
            for item in thelist:
                f.write("%s\n" % item)
    except Exception as e:
        print(f"Erro ao salvar lista em {path}: {e}")


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

    if not args.fileVocab or not args.fileModelConfig or not args.fileModel:
         raise ValueError("Caminhos para Vocab, Config ou Modelo não foram especificados.")
         
    model = MyModel(args).to(device)
    return model

def init_optim(args, model):
    """
    Inicializa o otimizador AdamW.
    """
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
    ]
    
    # verifica se há parâmetros treináveis
    has_trainable_params = any(p.requires_grad for group in optimizer_grouped_parameters for p in group['params'])
    if not has_trainable_params:
         print("Aviso: Nenhum parâmetro requer gradiente. Verifique a configuração de --numFreeze.")
         return AdamW([torch.nn.Parameter(torch.zeros(1))], lr=args.learning_rate) 

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    return optimizer

def init_lr_scheduler(args, optim):
    """
    Inicializa o agendador de taxa de aprendizado (learning rate scheduler).
    """
    t_total = args.epochs * args.episodeTrain
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    return scheduler


def deal_data(support_set, query_set, episode_internal_ids, labels_dict):
    """
    Processa os dados de um lote (episódio) para o formato que o modelo espera.
    Principalmente, converte os rótulos de texto para o formato one-hot.
    """
    text, original_labels = [], []
    # junta o texto e os rótulos originais do conjunto de suporte
    for x in support_set:
        text.append(x["text"])
        original_labels.append(x["label"]) # Ex: 'Negativo', 'Positivo'...
    # junta o texto e os rótulos originais do conjunto de consulta
    for x in query_set:
        text.append(x["text"])
        original_labels.append(x["label"]) # Ex: 'Negativo', 'Positivo'...
    
    one_hot_labels = []
    num_classes_in_episode = len(episode_internal_ids)
    # cria mapeamento reverso ID_interno -> índice_no_episódio (0 a k-1)
    internal_id_to_episode_index = {internal_id: idx for idx, internal_id in enumerate(episode_internal_ids)}

    for orig_label in original_labels:
        internal_id = labels_dict.get(orig_label) # mapeia label original ('Negativo') para ID interno (0)
        if internal_id is None or internal_id not in internal_id_to_episode_index:
             print(f"Aviso em deal_data: Label '{orig_label}' inválido ou não pertence a este episódio {episode_internal_ids}. Labels dict: {labels_dict}")
             one_hot = [-1] * num_classes_in_episode 
        else:
            episode_index = internal_id_to_episode_index[internal_id]
            one_hot = [0] * num_classes_in_episode
            one_hot[episode_index] = 1
        one_hot_labels.append(one_hot)

    return text, one_hot_labels

def train(args, tr_dataloader, model, optim, lr_scheduler, labels_dict, val_dataloader=None):
    """
    A função principal que executa o loop de treinamento e validação.
    """
    acc_best_state = None 
    
    epoch_train_loss, epoch_train_acc, epoch_train_p, epoch_train_r, epoch_train_f1, epoch_train_auc, epoch_train_topkacc = [], [], [], [], [], [], []
    epoch_val_loss, epoch_val_acc, epoch_val_p, epoch_val_r, epoch_val_f1, epoch_val_auc, epoch_val_topkacc = [], [], [], [], [], [], []
    
    best_acc = 0.0 
    best_p, best_r, best_f1, best_auc = 0.0, 0.0, 0.0, 0.0
    loss_fn = Loss_fn(args)
    acc_best_model_path = os.path.join(args.fileModelSave, 'acc_best_model.pth')
    cycle = 0 
    
    if labels_dict is None: 
        print("Erro: labels_dict não foi fornecido para a função train.")
        return None 

    id2label = {idx: original_label for original_label, idx in labels_dict.items()}
    device = next(model.parameters()).device # pega o device do modelo uma vez

    for epoch in range(args.epochs):
        print('=== Época: {} ==='.format(epoch))
        model.train() 
        
        if cycle >= args.patience:
            print(f"Parando cedo na época {epoch}.")
            break

        # listas para métricas dos batches DENTRO dessa época
        batch_train_loss, batch_train_acc, batch_train_p, batch_train_r, batch_train_f1, batch_train_auc, batch_train_topkacc = [], [], [], [], [], [], []

        for i, batch in tqdm(enumerate(tr_dataloader), total=args.episodeTrain, desc=f"Treino Época {epoch}"):
            optim.zero_grad()
            support_set, query_set, episode_internal_ids = batch # IDs: [0, 1, 2] etc.
            
            try:
                # usa id2label para converter IDs internos de volta para as strings corretas
                label_text = [id2label[int(el)] for el in episode_internal_ids] 
            except KeyError as e:
                 print(f"\nErro no treino ao gerar label_text: ID {e} não encontrado em id2label {id2label}. Episódio: {episode_internal_ids}")
                 label_text = [str(el) for el in episode_internal_ids] 
                 print(f"Usando IDs numéricos como fallback: {label_text}")

            text, one_hot_labels = deal_data(support_set, query_set, episode_internal_ids, labels_dict)
            
            # converte one_hot_labels para tensor
            one_hot_labels_tensor = torch.tensor(one_hot_labels, dtype=torch.float).to(device)

            # pula batch se houver labels inválidos
            if (one_hot_labels_tensor == -1).any():
                print("Aviso: Labels inválidos detectados no batch de treino. Pulando.")
                continue 

            try:
                 model_outputs = model(text, label_text) # passa textos e as STRINGS dos labels
                 loss, p, r, f, acc, auc, topk_acc = loss_fn(model_outputs, one_hot_labels_tensor)
            except Exception as e:
                 print(f"\nErro durante o forward/loss no treino (batch {i}): {e}")
                 print(f"  Input text (primeiros 10): {text[:10]}")
                 print(f"  Input label_text: {label_text}")
                 print(f"  Labels tensor shape: {one_hot_labels_tensor.shape}")
                 continue 

            if torch.isnan(loss) or torch.isinf(loss):
                 print(f"Aviso: Perda inválida (NaN ou Inf) no treino (batch {i}). Pulando backward.")
                 continue 

            loss.backward()
            optim.step()
            lr_scheduler.step()
            
            # guarda métricas do batch
            batch_train_loss.append(loss.item())
            batch_train_p.append(p)
            batch_train_r.append(r)
            batch_train_f1.append(f)
            batch_train_acc.append(acc)
            batch_train_auc.append(auc)
            batch_train_topkacc.append(topk_acc.item() if isinstance(topk_acc, torch.Tensor) else topk_acc)


        # calcula e imprime médias da época de treino
        avg_loss = np.mean(batch_train_loss) if batch_train_loss else 0
        avg_acc = np.mean(batch_train_acc) if batch_train_acc else 0
        avg_p = np.mean(batch_train_p) if batch_train_p else 0
        avg_r = np.mean(batch_train_r) if batch_train_r else 0
        avg_f1 = np.mean(batch_train_f1) if batch_train_f1 else 0
        avg_auc = np.mean(batch_train_auc) if batch_train_auc else 0
        avg_topkacc = np.mean(batch_train_topkacc) if batch_train_topkacc else 0
        print('\nMédia Train Época {}: Loss: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f}, Acc: {:.4f}, AUC: {:.4f}, TopKAcc: {:.4f}'.format(
            epoch, avg_loss, avg_p, avg_r, avg_f1, avg_acc, avg_auc, avg_topkacc))
        
        # guarda médias da época
        epoch_train_loss.append(avg_loss); epoch_train_acc.append(avg_acc); epoch_train_p.append(avg_p)
        epoch_train_r.append(avg_r); epoch_train_f1.append(avg_f1); epoch_train_auc.append(avg_auc)
        epoch_train_topkacc.append(avg_topkacc)

        # validação
        if val_dataloader is None:
             # salva o modelo da última época se não houver validação
             if epoch == args.epochs - 1: # salva apenas no final
                 print("Salvando modelo da última época (sem validação).")
                 torch.save(model.state_dict(), acc_best_model_path)
             continue 

        batch_val_loss, batch_val_acc, batch_val_p, batch_val_r, batch_val_f1, batch_val_auc, batch_val_topkacc = [], [], [], [], [], [], []

        with torch.no_grad():
            model.eval()
            
            for batch in tqdm(val_dataloader, total=args.episodeTrain, desc=f"Validação Época {epoch}"):
                support_set, query_set, episode_internal_ids = batch
                
                try:
                    label_text = [id2label[int(el)] for el in episode_internal_ids]
                except KeyError as e:
                    print(f"\nErro na validação ao gerar label_text: ID {e} não encontrado.")
                    label_text = [str(el) for el in episode_internal_ids]
                    print(f"Fallback validação: {label_text}")

                text, one_hot_labels = deal_data(support_set, query_set, episode_internal_ids, labels_dict)
                one_hot_labels_tensor = torch.tensor(one_hot_labels, dtype=torch.float).to(device)

                if (one_hot_labels_tensor == -1).any():
                     print("Aviso: Labels inválidos na validação. Pulando batch.")
                     continue
                
                try:
                    model_outputs = model(text, label_text)
                    loss, p, r, f, acc, auc, topkacc = loss_fn(model_outputs, one_hot_labels_tensor)
                except Exception as e:
                     print(f"\nErro durante o forward/loss na validação: {e}")
                     continue 

                batch_val_loss.append(loss.item())
                batch_val_acc.append(acc)
                batch_val_p.append(p)
                batch_val_r.append(r)
                batch_val_f1.append(f)
                batch_val_auc.append(auc)
                batch_val_topkacc.append(topkacc.item() if isinstance(topkacc, torch.Tensor) else topkacc)
            
            # calcula médias da época de validação
            avg_loss = np.mean(batch_val_loss) if batch_val_loss else 0
            avg_acc = np.mean(batch_val_acc) if batch_val_acc else 0
            avg_p = np.mean(batch_val_p) if batch_val_p else 0
            avg_r = np.mean(batch_val_r) if batch_val_r else 0
            avg_f1 = np.mean(batch_val_f1) if batch_val_f1 else 0
            avg_auc = np.mean(batch_val_auc) if batch_val_auc else 0
            avg_topkacc = np.mean(batch_val_topkacc) if batch_val_topkacc else 0
            
            # guarda médias da época
            epoch_val_loss.append(avg_loss); epoch_val_acc.append(avg_acc); epoch_val_p.append(avg_p)
            epoch_val_r.append(avg_r); epoch_val_f1.append(avg_f1); epoch_val_auc.append(avg_auc)
            epoch_val_topkacc.append(avg_topkacc)

            # lógica para salvar melhor modelo e early stopping
            acc_prefix = ' (Melhor)' if avg_acc > best_acc else ' (Melhor: {:.4f})'.format(best_acc)
            print('\nMédia Val Época {}: Loss: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f}, Acc: {:.4f}{}, AUC: {:.4f}, TopKAcc: {:.4f}'.format(
                epoch, avg_loss, avg_p, avg_r, avg_f1, avg_acc, acc_prefix, avg_auc, avg_topkacc))
            
            if avg_acc > best_acc:
                print(f"Nova melhor acurácia de validação: {avg_acc:.4f}. Salvando modelo...")
                torch.save(model.state_dict(), acc_best_model_path)
                best_acc = avg_acc
                acc_best_state = model.state_dict() # guarda o estado na memória também
                best_p, best_r, best_f1, best_auc = avg_p, avg_r, avg_f1, avg_auc
                cycle = 0 
            else:
                cycle += 1 
                

    # salva as métricas de todas as épocas
    metrics_to_save = {
        'epoch_train_loss': epoch_train_loss, 'epoch_train_p': epoch_train_p, 
        'epoch_train_r': epoch_train_r, 'epoch_train_f1': epoch_train_f1, 
        'epoch_train_acc': epoch_train_acc, 'epoch_train_auc': epoch_train_auc, 
        'epoch_train_topkacc': epoch_train_topkacc,
        'epoch_val_loss': epoch_val_loss, 'epoch_val_p': epoch_val_p, 
        'epoch_val_r': epoch_val_r, 'epoch_val_f1': epoch_val_f1, 
        'epoch_val_acc': epoch_val_acc, 'epoch_val_auc': epoch_val_auc, 
        'epoch_val_topkacc': epoch_val_topkacc
    }
    os.makedirs(args.fileModelSave, exist_ok=True) 
    for name, data_list in metrics_to_save.items():
        if data_list: 
             save_list_to_file(os.path.join(args.fileModelSave, name + '.txt'), data_list)

    print("Treinamento concluído.")
    # carrega o melhor estado salvo na memória ou do arquivo
    if acc_best_state is not None:
         print("Carregando o melhor estado do modelo (da validação) para teste.")
         model.load_state_dict(acc_best_state)
    elif os.path.exists(acc_best_model_path):
         print(f"Carregando o melhor modelo salvo de {acc_best_model_path}")
         try:
              model.load_state_dict(torch.load(acc_best_model_path, map_location=device))
         except Exception as e:
              print(f"Erro ao carregar o melhor modelo: {e}. Usando o modelo da última época.")
    else:
         print("Aviso: Nenhum modelo foi salvo. Usando o modelo da última época para teste.")
         
    return model

def test(args, test_dataloader, model, labels_dict):
    """ Testa o modelo treinado. """
    batch_test_loss, batch_test_acc, batch_test_p, batch_test_r, batch_test_f1, batch_test_auc, batch_test_topkacc = [], [], [], [], [], [], []
    loss_fn = Loss_fn(args)
    
    model.eval()
    
    if labels_dict is None: 
        print("Erro crítico no teste: Dicionário de rótulos não encontrado.")
        return 
        
    id2label = {idx: original_label for original_label, idx in labels_dict.items()}
    device = next(model.parameters()).device
    
    with torch.no_grad(): 
        for batch in tqdm(test_dataloader, total=args.episodeTest, desc="Teste"):
            support_set, query_set, episode_internal_ids = batch
            
            try:
                label_text = [id2label[int(el)] for el in episode_internal_ids]
            except KeyError as e:
                print(f"\nErro no teste ao gerar label_text: ID {e} não encontrado.")
                label_text = [str(el) for el in episode_internal_ids]
                print(f"Fallback teste: {label_text}")
            
            text, one_hot_labels = deal_data(support_set, query_set, episode_internal_ids, labels_dict)
            one_hot_labels_tensor = torch.tensor(one_hot_labels, dtype=torch.float).to(device)

            if (one_hot_labels_tensor == -1).any():
                 print("Aviso: Labels inválidos no teste. Pulando batch.")
                 continue

            try:
                 model_outputs = model(text, label_text)
                 loss, p, r, f, acc, auc, topkacc = loss_fn(model_outputs, one_hot_labels_tensor)
            except Exception as e:
                 print(f"\nErro durante o forward/loss no teste: {e}")
                 continue 

            # guarda métricas
            batch_test_loss.append(loss.item())
            batch_test_acc.append(acc)
            batch_test_p.append(p)
            batch_test_r.append(r)
            batch_test_f1.append(f)
            batch_test_auc.append(auc)
            batch_test_topkacc.append(topkacc.item() if isinstance(topkacc, torch.Tensor) else topkacc)
            
    # calcula médias do teste
    avg_loss = np.mean(batch_test_loss) if batch_test_loss else 0
    avg_acc = np.mean(batch_test_acc) if batch_test_acc else 0
    avg_p = np.mean(batch_test_p) if batch_test_p else 0
    avg_r = np.mean(batch_test_r) if batch_test_r else 0
    avg_f1 = np.mean(batch_test_f1) if batch_test_f1 else 0
    avg_auc = np.mean(batch_test_auc) if batch_test_auc else 0
    avg_topkacc = np.mean(batch_test_topkacc) if batch_test_topkacc else 0

    print('\nResultados Finais do Teste')
    print(f'Test Loss: {avg_loss:.4f}')
    print(f'Test Acc: {avg_acc:.4f}')
    print(f'Test P (macro): {avg_p:.4f}')
    print(f'Test R (macro): {avg_r:.4f}')
    print(f'Test F1 (macro): {avg_f1:.4f}')
    print(f'Test AUC: {avg_auc:.4f}')
    print(f'Test TopKAcc: {avg_topkacc:.4f}') 

    os.makedirs(args.fileModelSave, exist_ok=True) 
    result_csv_path = os.path.join(args.fileModelSave, 'result.csv')
    try:
        write_header = not os.path.exists(result_csv_path)
        with open(result_csv_path, 'a+', newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            data = [
                "commont", args.commont, 
                "data", args.dataFile,
                "shot", args.numKShot, 
                "acc", avg_acc 
            ]
            writer.writerow(data)
    except Exception as e: print(f"Erro ao escrever no result.csv: {e}")

    test_score_json_path = os.path.join(args.fileModelSave, "test_score.json")
    try:
        tmp = {
            "commont": args.commont, "data": args.dataFile, "shot": args.numKShot, 
            "acc": avg_acc, "p": avg_p, "r": avg_r, "f1": avg_f1, 
            "auc": avg_auc, "Loss": avg_loss, "topkacc": avg_topkacc
        }
        with open(test_score_json_path, "a+", encoding='utf-8') as fout:
            fout.write("%s\n" % json.dumps(tmp, ensure_ascii=False))
    except Exception as e: print(f"Erro ao escrever no test_score.json: {e}")

def write_args_to_json(args): 
    """ Salva os argumentos do experimento em config.json. """
    path = os.path.join(args.fileModelSave, "config.json")
    try:
        os.makedirs(args.fileModelSave, exist_ok=True) 
        args_dict = vars(args) 
        safe_args = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in args_dict.items()}

        safe_args = {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v for k, v in safe_args.items()}
        json_str = json.dumps(safe_args, indent=4) 
        with open(path, 'w', encoding='utf-8') as json_file: 
            json_file.write(json_str)
    except Exception as e: print(f"Erro ao salvar config.json: {e}")

def main():
    """ Função principal. """
    args = get_parser().parse_args()

    try: os.makedirs(args.fileModelSave, exist_ok=True)
    except OSError as e: print(f"Erro ao criar diretório {args.fileModelSave}: {e}"); return

    write_args_to_json(args)
    set_seed(args.seed if hasattr(args, 'seed') else 42) 
    
    try:
        labels_dict = get_label_dict(args) 
        if labels_dict is None: 
            raise ValueError("Falha ao criar o dicionário de rótulos a partir do train.json.")

        model = init_model(args)
        if model is None: raise ValueError("Falha ao inicializar o modelo.")
        
        print("\nInicializando Dataloader de Treino...")
        tr_sampler = init_dataloader(args, 'train', labels_dict)
        
        print("\nInicializando Dataloader de Validação...")
        val_sampler = init_dataloader(args, 'valid', labels_dict)
        
        print("\nInicializando Dataloader de Teste...")
        test_sampler = init_dataloader(args, 'test', labels_dict)
        
        # trata o caso de valid/test não existirem ou estarem vazios
        if tr_sampler is None:
             raise ValueError("Dataloader de treino não pôde ser criado ou está vazio.")
        if val_sampler is None:
            print("Aviso: Dataloader de validação não foi criado. O modelo não será validado por época.")
        if test_sampler is None:
            print("Aviso: Dataloader de teste não foi criado. O modelo não será testado no final.")

        optim = init_optim(args, model)
        if optim is None: raise ValueError("Falha ao inicializar o otimizador.")
        
        lr_scheduler = init_lr_scheduler(args, optim)
    except Exception as e:
        print(f"Erro fatal durante a inicialização: {e}")
        return

    trained_model = train(args=args,
                          tr_dataloader=tr_sampler, 
                          val_dataloader=val_sampler, 
                          model=model,
                          optim=optim,
                          lr_scheduler=lr_scheduler,
                          labels_dict=labels_dict) 
    
    if trained_model is None:
         print("Treinamento falhou ou foi abortado. Encerrando.")
         return

    if test_sampler:
        print('\nTestando com o melhor modelo...')
        test(args=args,
             test_dataloader=test_sampler, 
             model=trained_model,
             labels_dict=labels_dict) 
    else:
        print("\nNenhum dataset de teste para rodar. Encerrando.")

if __name__ == '__main__':
    main()