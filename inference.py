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

BASE_RESULTS = "/raid/user_beatrizalmeida/qevasion/QEvasion_Augmented_Results"
OFFICIAL_DATA = "data/Official_Submission" 
LOCAL_MODEL = "/raid/user_beatrizalmeida/models/bge-m3"
ORIGINAL_TRAIN_BASE = "data/QEvasionCorpus/few_shot" 

FOLDERS = [
    "kshot_55_fold_01", "kshot_55_fold_02", "kshot_55_fold_03", 
    "kshot_55_fold_04", "kshot_55_fold_05"
]

def load_args(path):
    with open(path, 'r') as f: return SimpleNamespace(**json.load(f))

def predict(fold_name, gpu_id=0):
    print(f"\n>>> Processando: {fold_name}")
    fold_path = os.path.join(BASE_RESULTS, fold_name)
    
    args = load_args(os.path.join(fold_path, "config.json"))
    args.numDevice = gpu_id
    args.fileModelSave = fold_path 
    
    args.dataFile = OFFICIAL_DATA 
    
    if os.path.exists(LOCAL_MODEL):
        args.fileModel = LOCAL_MODEL
        args.fileVocab = LOCAL_MODEL
        args.fileModelConfig = LOCAL_MODEL

    device = torch.device('cuda', args.numDevice) if torch.cuda.is_available() else torch.device('cpu')

    fold_id = fold_name.split("_fold_")[-1]
    path_to_original_train = os.path.join(ORIGINAL_TRAIN_BASE, fold_id)
    
    args_temp = SimpleNamespace(**vars(args))
    args_temp.dataFile = path_to_original_train
    labels_dict = get_label_dict(args_temp)
    
    if labels_dict is None: return

    model = init_model(args)
    model.load_state_dict(torch.load(os.path.join(fold_path, "acc_best_model.pth"), map_location=device))
    model.to(device)

    print("Carregando teste oficial...")
    test_sampler = init_dataloader(args, 'test', labels_dict)

    test(args, test_sampler, model, labels_dict)
    
    old_file = os.path.join(fold_path, "predictions.txt")
    new_file = os.path.join(fold_path, "official_submission_preds.txt")
    if os.path.exists(old_file):
        os.rename(old_file, new_file)
        print(f"Salvo em: {new_file}")

if __name__ == "__main__":
    for fold in FOLDERS:
        predict(fold)