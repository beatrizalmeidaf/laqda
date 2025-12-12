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

def set_seed(seed):
    """
    Define as seeds para todas as bibliotecas de aleatoriedade
    para garantir que os resultados do experimento sejam reprodutíveis.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def init_dataloader(args, mode, labels_dict):
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
    """
    text, original_labels = [], []
    for x in support_set:
        text.append(x["text"])
        original_labels.append(x["label"]) 
    for x in query_set:
        text.append(x["text"])
        original_labels.append(x["label"]) 
    
    one_hot_labels = []
    num_classes_in_episode = len(episode_internal_ids)
    internal_id_to_episode_index = {internal_id: idx for idx, internal_id in enumerate(episode_internal_ids)}

    for orig_label in original_labels:
        internal_id = labels_dict.get(orig_label) 
        if internal_id is None or internal_id not in internal_id_to_episode_index:
             one_hot = [-1] * num_classes_in_episode 
        else:
            episode_index = internal_id_to_episode_index[internal_id]
            one_hot = [0] * num_classes_in_episode
            one_hot[episode_index] = 1
        one_hot_labels.append(one_hot)

    return text, one_hot_labels

def save_predictions_to_file(args, predictions):
    """
    Salva a lista de predições em um arquivo 'predictions.txt'.
    """
    predictions_path = os.path.join(args.fileModelSave, 'predictions.txt')
    try:
        os.makedirs(args.fileModelSave, exist_ok=True)
        # 'w' sobrescreve o arquivo se ele já existir
        with open(predictions_path, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(f"{pred}\n")
        print(f"Previsões salvas em: {predictions_path}")
    except Exception as e:
        print(f"Erro ao salvar previsões: {e}")

def train(args, tr_dataloader, model, optim, lr_scheduler, labels_dict, val_dataloader=None):
    """
    Loop de treinamento.
    """
    acc_best_state = None 
    
    epoch_train_loss, epoch_train_acc, epoch_train_p, epoch_train_r, epoch_train_f1, epoch_train_auc, epoch_train_topkacc = [], [], [], [], [], [], []
    epoch_val_loss, epoch_val_acc, epoch_val_p, epoch_val_r, epoch_val_f1, epoch_val_auc, epoch_val_topkacc = [], [], [], [], [], [], []
    
    best_acc = 0.0 
    loss_fn = Loss_fn(args)
    acc_best_model_path = os.path.join(args.fileModelSave, 'acc_best_model.pth')
    cycle = 0 
    
    if labels_dict is None: 
        print("Erro: labels_dict não foi fornecido para a função train.")
        return None 

    id2label = {idx: original_label for original_label, idx in labels_dict.items()}
    device = next(model.parameters()).device 

    for epoch in range(args.epochs):
        print('=== Época: {} ==='.format(epoch))
        model.train() 
        
        if cycle >= args.patience:
            print(f"Parando cedo na época {epoch}.")
            break

        batch_train_loss, batch_train_acc, batch_train_p, batch_train_r, batch_train_f1, batch_train_auc, batch_train_topkacc = [], [], [], [], [], [], []

        for i, batch in tqdm(enumerate(tr_dataloader), total=args.episodeTrain, desc=f"Treino Época {epoch}"):
            optim.zero_grad()
            support_set, query_set, episode_internal_ids = batch
            
            try:
                label_text = [id2label[int(el)] for el in episode_internal_ids] 
            except KeyError as e:
                 label_text = [str(el) for el in episode_internal_ids] 

            text, one_hot_labels = deal_data(support_set, query_set, episode_internal_ids, labels_dict)
            one_hot_labels_tensor = torch.tensor(one_hot_labels, dtype=torch.float).to(device)

            if (one_hot_labels_tensor == -1).any():
                continue 

            try:
                 model_outputs = model(text, label_text)
                 loss, p, r, f, acc, auc, topk_acc = loss_fn(model_outputs, one_hot_labels_tensor)
            except Exception as e:
                 print(f"Erro treino batch {i}: {e}")
                 continue 

            if torch.isnan(loss) or torch.isinf(loss):
                 continue 

            loss.backward()
            optim.step()
            lr_scheduler.step()
            
            batch_train_loss.append(loss.item())
            batch_train_p.append(p)
            batch_train_r.append(r)
            batch_train_f1.append(f)
            batch_train_acc.append(acc)
            batch_train_auc.append(auc)
            batch_train_topkacc.append(topk_acc.item() if isinstance(topk_acc, torch.Tensor) else topk_acc)

        avg_loss = np.mean(batch_train_loss) if batch_train_loss else 0
        avg_acc = np.mean(batch_train_acc) if batch_train_acc else 0
        avg_p = np.mean(batch_train_p) if batch_train_p else 0
        avg_r = np.mean(batch_train_r) if batch_train_r else 0
        avg_f1 = np.mean(batch_train_f1) if batch_train_f1 else 0
        avg_auc = np.mean(batch_train_auc) if batch_train_auc else 0
        avg_topkacc = np.mean(batch_train_topkacc) if batch_train_topkacc else 0
        print('\nMédia Train Época {}: Loss: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f}, Acc: {:.4f}, AUC: {:.4f}, TopKAcc: {:.4f}'.format(
            epoch, avg_loss, avg_p, avg_r, avg_f1, avg_acc, avg_auc, avg_topkacc))
        
        epoch_train_loss.append(avg_loss); epoch_train_acc.append(avg_acc); epoch_train_p.append(avg_p)
        epoch_train_r.append(avg_r); epoch_train_f1.append(avg_f1); epoch_train_auc.append(avg_auc)
        epoch_train_topkacc.append(avg_topkacc)

        if val_dataloader is None:
             if epoch == args.epochs - 1:
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
                    label_text = [str(el) for el in episode_internal_ids]

                text, one_hot_labels = deal_data(support_set, query_set, episode_internal_ids, labels_dict)
                one_hot_labels_tensor = torch.tensor(one_hot_labels, dtype=torch.float).to(device)

                if (one_hot_labels_tensor == -1).any():
                     continue
                
                try:
                    model_outputs = model(text, label_text)
                    loss, p, r, f, acc, auc, topk_acc = loss_fn(model_outputs, one_hot_labels_tensor)
                except Exception as e:
                     continue 

                batch_val_loss.append(loss.item())
                batch_val_acc.append(acc)
                batch_val_p.append(p)
                batch_val_r.append(r)
                batch_val_f1.append(f)
                batch_val_auc.append(auc)
                batch_val_topkacc.append(topk_acc.item() if isinstance(topk_acc, torch.Tensor) else topk_acc)
            
            avg_loss = np.mean(batch_val_loss) if batch_val_loss else 0
            avg_acc = np.mean(batch_val_acc) if batch_val_acc else 0
            avg_p = np.mean(batch_val_p) if batch_val_p else 0
            avg_r = np.mean(batch_val_r) if batch_val_r else 0
            avg_f1 = np.mean(batch_val_f1) if batch_val_f1 else 0
            avg_auc = np.mean(batch_val_auc) if batch_val_auc else 0
            avg_topkacc = np.mean(batch_val_topkacc) if batch_val_topkacc else 0
            
            epoch_val_loss.append(avg_loss); epoch_val_acc.append(avg_acc); epoch_val_p.append(avg_p)
            epoch_val_r.append(avg_r); epoch_val_f1.append(avg_f1); epoch_val_auc.append(avg_auc)
            epoch_val_topkacc.append(avg_topkacc)

            acc_prefix = ' (Melhor)' if avg_acc > best_acc else ' (Melhor: {:.4f})'.format(best_acc)
            print('\nMédia Val Época {}: Loss: {:.4f}, P: {:.4f}, R: {:.4f}, F1: {:.4f}, Acc: {:.4f}{}, AUC: {:.4f}, TopKAcc: {:.4f}'.format(
                epoch, avg_loss, avg_p, avg_r, avg_f1, avg_acc, acc_prefix, avg_auc, avg_topkacc))
            
            if avg_acc > best_acc:
                print(f"Nova melhor acurácia de validação: {avg_acc:.4f}. Salvando modelo...")
                torch.save(model.state_dict(), acc_best_model_path)
                best_acc = avg_acc
                acc_best_state = model.state_dict()
                cycle = 0 
            else:
                cycle += 1 

    metrics_to_save = {
        'epoch_train_loss': epoch_train_loss, 'epoch_train_acc': epoch_train_acc, 
        'epoch_val_loss': epoch_val_loss, 'epoch_val_acc': epoch_val_acc, 
        'epoch_val_f1': epoch_val_f1
    }
    os.makedirs(args.fileModelSave, exist_ok=True) 
    for name, data_list in metrics_to_save.items():
        if data_list: 
             save_list_to_file(os.path.join(args.fileModelSave, name + '.txt'), data_list)

    print("Treinamento concluído.")
    if acc_best_state is not None:
         model.load_state_dict(acc_best_state)
    elif os.path.exists(acc_best_model_path):
         try:
              model.load_state_dict(torch.load(acc_best_model_path, map_location=device))
         except Exception: pass
         
    return model

def test(args, test_dataloader, model, labels_dict):
    """ Testa o modelo treinado. """
    batch_test_loss, batch_test_acc, batch_test_p, batch_test_r, batch_test_f1, batch_test_auc, batch_test_topkacc = [], [], [], [], [], [], []
    loss_fn = Loss_fn(args)
    
    all_predicted_labels = [] 
    
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
                 
                 prototypes, query_embeddings, _, _, _ = model_outputs

                 dists = torch.pow(query_embeddings.unsqueeze(1) - prototypes.unsqueeze(0), 2).sum(2)
                 
                 predicted_indices = torch.argmin(dists, dim=1).cpu().numpy()
                 
                 for idx in predicted_indices:
                     if idx < len(label_text):
                         all_predicted_labels.append(label_text[idx])
                     else:
                         all_predicted_labels.append("UNKNOWN")

                 loss, p, r, f, acc, auc, topk_acc = loss_fn(model_outputs, one_hot_labels_tensor)

            except Exception as e:
                 print(f"\nErro durante o forward/loss no teste: {e}")
                 continue 

            batch_test_loss.append(loss.item())
            batch_test_acc.append(acc)
            batch_test_p.append(p)
            batch_test_r.append(r)
            batch_test_f1.append(f)
            batch_test_auc.append(auc)
            batch_test_topkacc.append(topk_acc.item() if isinstance(topk_acc, torch.Tensor) else topk_acc)
            
    save_predictions_to_file(args, all_predicted_labels)

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
    print(f'Total de previsões salvas: {len(all_predicted_labels)}')

    os.makedirs(args.fileModelSave, exist_ok=True) 
    result_csv_path = os.path.join(args.fileModelSave, 'result.csv')
    try:
        with open(result_csv_path, 'a+', newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            data = [
                "commont", args.commont, 
                "data", args.dataFile,
                "shot", args.numKShot, 
                "acc", avg_acc,
                "f1", avg_f1 
            ]
            writer.writerow(data)
    except Exception as e: print(f"Erro ao escrever no result.csv: {e}")

def write_args_to_json(args): 
    path = os.path.join(args.fileModelSave, "config.json")
    try:
        os.makedirs(args.fileModelSave, exist_ok=True) 
        args_dict = vars(args) 
        safe_args = {k: str(v) for k, v in args_dict.items()}
        json_str = json.dumps(safe_args, indent=4) 
        with open(path, 'w', encoding='utf-8') as json_file: 
            json_file.write(json_str)
    except Exception: pass

def main():
    args = get_parser().parse_args()
    try: os.makedirs(args.fileModelSave, exist_ok=True)
    except OSError: return

    write_args_to_json(args)
    set_seed(args.seed if hasattr(args, 'seed') else 42) 
    
    try:
        labels_dict = get_label_dict(args) 
        if labels_dict is None: raise ValueError("Erro label dict.")

        model = init_model(args)
        
        print("\nInicializando Dataloader de Treino...")
        tr_sampler = init_dataloader(args, 'train', labels_dict)
        val_sampler = init_dataloader(args, 'valid', labels_dict)
        test_sampler = init_dataloader(args, 'test', labels_dict)
        
        optim = init_optim(args, model)
        lr_scheduler = init_lr_scheduler(args, optim)
    except Exception as e:
        print(f"Erro inicialização: {e}")
        return

    trained_model = train(args=args, tr_dataloader=tr_sampler, val_dataloader=val_sampler, 
                          model=model, optim=optim, lr_scheduler=lr_scheduler, labels_dict=labels_dict) 
    
    if test_sampler and trained_model:
        print('\nTestando com o melhor modelo...')
        test(args=args, test_dataloader=test_sampler, model=trained_model, labels_dict=labels_dict) 

if __name__ == '__main__':
    main()