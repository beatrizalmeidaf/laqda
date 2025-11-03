import json
import os
from pathlib import Path
from tqdm import tqdm


ORIGINAL_DATA_BASE_PATH = Path("datasets-br/reviews") 

REMAPPED_DATA_BASE_PATH = Path("datasets-br/reviews_remapped_sentiment_binary") 

DATASETS_TO_PROCESS = ["B2WCorpus", "BrandsCorpus", "ReProCorpus", "UTLCorpus"]

LABEL_MAP = {
    
    1: "Negativo",
    2: "Negativo",
    #3: "Neutro",
    4: "Positivo",
    5: "Positivo"
}


def remap_labels_in_file(input_path: Path, output_path: Path):
    """
    Lê um arquivo JSON contendo uma LISTA de objetos, remapeia o campo 'label' 
    e salva em um novo arquivo no formato JSONL (JSON Lines).
    """
    print(f"Processando: {input_path} -> {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    error_count = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            try:
                original_data = json.load(infile) # carrega a lista inteira
                if not isinstance(original_data, list):
                    print(f"  Erro: O arquivo {input_path} não contém uma lista JSON válida.")
                    return 
            except json.JSONDecodeError as e:
                print(f"  Erro: Falha ao decodificar o JSON principal em {input_path}. Erro: {e}")
                return 

        with open(output_path, 'w', encoding='utf-8') as outfile:
            for data in tqdm(original_data, desc=f"  Processando {input_path.name}", leave=False):
                try:

                    if not isinstance(data, dict):
                        print(f"    Aviso: Item não é um dicionário JSON em {input_path}. Item: {data}")
                        error_count += 1
                        continue

                    original_label = data.get("label")
                    text = data.get("text") 

                    if text is None or original_label is None:
                        print(f"    Aviso: Registro sem 'text'/'sentence' ou 'label' em {input_path}. Registro: {data}")
                        error_count += 1
                        continue

                    # tenta converter o label original para int
                    try:
                        original_label_int = int(original_label)
                    except (ValueError, TypeError):
                         print(f"    Aviso: Label '{original_label}' não é um inteiro válido em {input_path}. Registro: {data}")
                         error_count += 1
                         continue

                    # aplica o mapeamento
                    new_label = LABEL_MAP.get(original_label_int)
                    
                    if new_label is None:
                        print(f"    Aviso: Label original {original_label_int} não encontrado no mapeamento. Registro: {data}")
                        error_count += 1
                        continue
                        
                    # cria o novo registro com o label remapeado
                    new_record = {"sentence": text, "label": new_label} 
                    
                    # escreve o novo registro no arquivo de saída (formato JSONL)
                    outfile.write(json.dumps(new_record, ensure_ascii=False) + "\n")
                    processed_count += 1

                except Exception as e:
                    print(f"    Erro inesperado processando registro: {e}. Registro: {data}")
                    error_count +=1

        print(f"  Concluído: {processed_count} linhas processadas, {error_count} erros/avisos.")
        
    except FileNotFoundError:
        print(f"Erro: Arquivo de entrada não encontrado: {input_path}")
    except Exception as e:
        print(f"Erro inesperado ao processar o arquivo {input_path}: {e}")


if __name__ == "__main__":
    print(f"Iniciando remapeamento de labels...")
    print(f"Datasets Originais em: {ORIGINAL_DATA_BASE_PATH.resolve()}")
    print(f"Datasets Remapeados serão salvos em: {REMAPPED_DATA_BASE_PATH.resolve()}")
    print("-" * 30)

    for dataset_name in DATASETS_TO_PROCESS:
        print(f"Processando Dataset: {dataset_name}")
        original_dataset_path = ORIGINAL_DATA_BASE_PATH / dataset_name / "few_shot"
        remapped_dataset_path = REMAPPED_DATA_BASE_PATH / dataset_name / "few_shot"

        if not original_dataset_path.is_dir():
            print(f"  Aviso: Diretório {original_dataset_path} não encontrado. Pulando dataset.")
            continue

        # itera sobre os folds (01 a 05)
        for fold_num in range(1, 6):
            fold_str = f"{fold_num:02d}" 
            original_fold_path = original_dataset_path / fold_str
            remapped_fold_path = remapped_dataset_path / fold_str
            
            print(f"  Processando Fold: {fold_str}")

            if not original_fold_path.is_dir():
                print(f"    Aviso: Diretório {original_fold_path} não encontrado. Pulando fold.")
                continue

            for split in ["train", "valid", "test"]:
                input_file = original_fold_path / f"{split}.json"
                output_file = remapped_fold_path / f"{split}.json"
                
                if input_file.exists():
                    remap_labels_in_file(input_file, output_file)
                else:
                    print(f"    Aviso: Arquivo {input_file} não encontrado.")

    
    print("Remapeamento de todos os datasets concluído!")