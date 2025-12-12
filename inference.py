import os
import torch
import json
import sys
from types import SimpleNamespace

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.main import init_model, test, init_dataloader
from src.data_loader import get_label_dict

BASE_RESULTS_DIR = "/raid/user_beatrizalmeida/qevasion/QEvasion_Augmented_Results"

# CAMINHO ABSOLUTO PARA OS DADOS (Corrigido conforme sua estrutura de pastas)
BASE_DATA_DIR = "/home/user_beatrizalmeida/laqda/data/QEvasionCorpus/few_shot"

FOLDERS_TO_PROCESS = [
    "kshot_55_fold_01",
    "kshot_55_fold_02",
    "kshot_55_fold_03",
    "kshot_55_fold_04",
    "kshot_55_fold_05"
]

def load_args_from_json(config_path):
    """Lê o config.json e converte para um objeto compatível com 'args'."""
    with open(config_path, 'r') as f:
        args_dict = json.load(f)
    return SimpleNamespace(**args_dict)

def run_inference_for_fold(fold_name, gpu_id=0):
    print(f"\n==================================================")
    print(f"Iniciando Inferência para: {fold_name}")
    
    # 1. Definir caminhos
    fold_path = os.path.join(BASE_RESULTS_DIR, fold_name)
    config_path = os.path.join(fold_path, "config.json")
    model_path = os.path.join(fold_path, "acc_best_model.pth")

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        print(f"[ERRO] Arquivos não encontrados em {fold_path}. Pulando.")
        return

    # 2. Carregar Configurações do Treino
    args = load_args_from_json(config_path)
    
    # --- AJUSTES PARA INFERÊNCIA ---
    args.numDevice = gpu_id
    args.fileModelSave = fold_path # Garante que predictions.txt salve na mesma pasta
    
    # Descobrir qual o ID do fold (ex: "01" de "kshot_55_fold_01")
    try:
        fold_id = fold_name.split("_fold_")[-1] # Pega o que vem depois de _fold_
        # Monta o caminho completo para a pasta do fold (ex: .../few_shot/01)
        args.dataFile = os.path.join(BASE_DATA_DIR, fold_id)
        print(f"Dataset configurado para: {args.dataFile}")
        
        # Verificação extra de segurança
        if not os.path.exists(os.path.join(args.dataFile, 'train.json')):
             print(f"[ERRO FATAL] O arquivo train.json não foi encontrado em: {args.dataFile}")
             print("Verifique se BASE_DATA_DIR está correto.")
             return

    except IndexError:
        print(f"[AVISO] Não foi possível extrair ID do fold do nome '{fold_name}'. Usando dataFile original: {args.dataFile}")

    # 3. Configurar Dispositivo (GPU)
    if torch.cuda.is_available():
        device = torch.device('cuda', args.numDevice)
        torch.cuda.set_device(device)
        print(f"Usando GPU: {args.numDevice}")
    else:
        device = torch.device('cpu')
        print("Usando CPU")

    # 4. Recriar Dicionário de Rótulos (Label Mapping)
    print("Recriando dicionário de rótulos...")
    labels_dict = get_label_dict(args)
    if labels_dict is None:
        print("[ERRO CRÍTICO] Falha ao criar labels_dict. Abortando este fold.")
        return

    # 5. Inicializar Modelo e Carregar Pesos Treinados
    print("Carregando modelo...")
    model = init_model(args)
    
    print(f"Carregando pesos de: {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"[ERRO] Falha ao carregar pesos: {e}")
        return
        
    model.to(device)

    # 6. Carregar Dados de Teste
    print("Carregando dataset de teste...")
    test_sampler = init_dataloader(args, 'test', labels_dict)
    
    if test_sampler is None:
        print("[ERRO] Dataloader de teste vazio.")
        return

    # 7. Executar Teste e Salvar Predições
    # Esta função (que vem do seu main.py modificado) já chama save_predictions_to_file
    print("Gerando predições...")
    test(args, test_sampler, model, labels_dict)
    
    print(f"Sucesso! Predições salvas em: {os.path.join(fold_path, 'predictions.txt')}")

def main():
    # Defina a GPU que deseja usar
    GPU_ID = 0 
    
    for folder in FOLDERS_TO_PROCESS:
        try:
            run_inference_for_fold(folder, gpu_id=GPU_ID)
        except Exception as e:
            print(f"[FALHA GERAL] Erro ao processar {folder}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()