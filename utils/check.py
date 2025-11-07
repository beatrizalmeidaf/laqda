import os
import json
from tqdm import tqdm

DATASETS_TO_PROCESS = [
    #"B2WCorpus", 
    #"BrandsCorpus", 
    "BuscapeCorpus", 
    "KaggleTweetsCorpus", 
    "OlistCorpus", 
    #"ReProCorpus", 
    #"UTLCorpus"
]
BASE_PATH = "datasets-br/sentiment_analysis_remapped" 
FOLDS = ["01", "02", "03", "04", "05"]


def load_jsonl_sentences(path):
    """Carrega apenas as sentenças de um arquivo JSONL, já normalizadas."""
    sentences = set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    sentence = data.get("sentence")
                    if sentence is not None:
                        # normaliza para garantir uma comparação justa
                        sentences.add(sentence.strip().lower()) 
                except json.JSONDecodeError:
                    print(f"  Aviso: Linha mal formatada em {os.path.basename(path)}, pulando.")
    except FileNotFoundError:
        print(f"  Aviso: Arquivo não encontrado {path}, pulando.")
        return None 
    except Exception as e:
        print(f"  Erro ao ler {path}: {e}")
        return None
    
    return sentences

def main():
    print("Iniciando verificação de contaminação de dados...")
    
    contaminated_datasets = []
    clean_datasets = []
    
    for dataset_name in DATASETS_TO_PROCESS:
        print(f"\n=======================================================")
        print(f"Verificando Dataset: {dataset_name}")
        print(f"=======================================================")
        
        dataset_base_path = os.path.join(BASE_PATH, dataset_name, "few_shot")
        is_dataset_contaminated = False # flag para o dataset inteiro
        
        for fold in FOLDS:
            fold_path = os.path.join(dataset_base_path, fold)
            
            # carregar os conjuntos de sentenças
            train_sents = load_jsonl_sentences(os.path.join(fold_path, "train.json"))
            valid_sents = load_jsonl_sentences(os.path.join(fold_path, "valid.json"))
            test_sents = load_jsonl_sentences(os.path.join(fold_path, "test.json"))

            if train_sents is None or valid_sents is None or test_sents is None:
                print(f"  Fold {fold}: AVISO - Faltando train.json, valid.json ou test.json. Pulando esse fold.")
                continue

            # verificar sobreposição (interseção)
            # a verificação mais ccritica é entre treino e teste
            tt_overlap = train_sents.intersection(test_sents)
            tv_overlap = train_sents.intersection(valid_sents)
            vt_overlap = valid_sents.intersection(test_sents)

            fold_is_contaminated = False
            
            if tt_overlap:
                print(f"  Fold {fold}: CONTAMINADO! {len(tt_overlap)} sentenças em comum entre 'train' e 'test'.")
                fold_is_contaminated = True
                
            if tv_overlap:
                print(f"  Fold {fold}: CONTAMINADO! {len(tv_overlap)} sentenças em comum entre 'train' e 'valid'.")
                fold_is_contaminated = True
                
            if vt_overlap:
                print(f"  Fold {fold}: CONTAMINADO! {len(vt_overlap)} sentenças em comum entre 'valid' e 'test'.")
                fold_is_contaminated = True

            if fold_is_contaminated:
                is_dataset_contaminated = True
            else:
                print(f"  Fold {fold}: LIMPO. Nenhuma sobreposição detectada.")
        
        # adicionar aos relatórios
        if is_dataset_contaminated:
            contaminated_datasets.append(dataset_name)
        else:
            clean_datasets.append(dataset_name)


    print("Verificação Concluída. Relatório Final:")
    
    print("\nDatasets LIMPOS:")
    if clean_datasets:
        for ds in clean_datasets:
            print(f"  - {ds}")
    else:
        print("  (Nenhum dataset 100% limpo em todos os folds)")
        
    print("\nDatasets CONTAMINADOS:")
    if contaminated_datasets:
        for ds in contaminated_datasets:
            print(f"  - {ds}")
    else:
        print("  (Nenhum dataset contaminado encontrado)")

if __name__ == "__main__":
    main()