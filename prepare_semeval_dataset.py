import pandas as pd
import os
import json
import shutil
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

INPUT_CSV = "augmented_dataset.csv"
OUTPUT_DIR = "data/QEvasionCorpus"
LABEL_COLUMN = "evasion_label"

# colunas de entrada
INPUT_TEXT_COLUMN_1 = 'interview_question'
INPUT_TEXT_COLUMN_2 = 'interview_answer'

# colunas de saída (padrão do dataloader)
FINAL_TEXT_COLUMN = 'sentence'
FINAL_LABEL_COLUMN = 'label'


# salvar em JSON Lines (JSONL) 
def save_as_jsonl(dataframe, file_path):
    """Salva o dataframe no formato JSONL que o data_loader.py espera."""
    print(f"Salvando {len(dataframe)} linhas em {file_path}...")
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True) 
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in dataframe.to_dict(orient='records'):
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f"Iniciando pré-processamento de {INPUT_CSV}")

output_path = Path(OUTPUT_DIR)
if output_path.exists():
    print(f"Limpando diretório de saída existente: {output_path}")
    shutil.rmtree(output_path)
os.makedirs(output_path, exist_ok=True)

try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"ERRO: Arquivo '{INPUT_CSV}' não encontrado.")
    exit()

print(f"Carregado {len(df)} linhas do CSV.")

df = df.dropna(subset=[INPUT_TEXT_COLUMN_1, INPUT_TEXT_COLUMN_2, LABEL_COLUMN])

# formata a entrada (sentence)
df[FINAL_TEXT_COLUMN] = df[INPUT_TEXT_COLUMN_1].astype(str) + " [SEP] " + df[INPUT_TEXT_COLUMN_2].astype(str)

df = df.rename(columns={LABEL_COLUMN: FINAL_LABEL_COLUMN})

df_final = df[[FINAL_TEXT_COLUMN, FINAL_LABEL_COLUMN]]

print(f"Total de {len(df_final)} exemplos válidos após limpeza inicial.")

print(f"\nTotal de amostras antes da deduplicação: {len(df_final)}")
df_deduplicated = df_final.drop_duplicates(subset=[FINAL_TEXT_COLUMN]).reset_index(drop=True)
removidas = len(df_final) - len(df_deduplicated)
print(f"Total de amostras únicas (após deduplicação): {len(df_deduplicated)} (removidas {removidas} duplicatas)")


print("Dividindo dados (80% treino, 10% validação, 10% teste) com estratificação...")
labels = df_deduplicated[FINAL_LABEL_COLUMN]
# 80% para treino e 20% para o restante
train_data, temp_data = train_test_split(df_deduplicated, 
                                         test_size=0.2, 
                                         random_state=42, 
                                         stratify=labels)

# divide o restante (20%) ao meio para validação e teste (10% cada)
valid_labels = temp_data[FINAL_LABEL_COLUMN]
valid_data, test_data = train_test_split(temp_data, 
                                         test_size=0.5, 
                                         random_state=42, 
                                         stratify=valid_labels)

save_as_jsonl(train_data, os.path.join(OUTPUT_DIR, 'train.json'))
save_as_jsonl(valid_data, os.path.join(OUTPUT_DIR, 'valid.json'))
save_as_jsonl(test_data, os.path.join(OUTPUT_DIR, 'test.json'))

print("\nPré-processamento concluído!")