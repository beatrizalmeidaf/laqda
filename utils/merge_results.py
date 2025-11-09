import pandas as pd
import os

base_dir = "/raid/user_beatrizalmeida/laqda_categ_results_br"

datasets = [
    "MMLU_PTBR_Corpus",
    "RecognasummCorpus",
    "RulingBRCorpus"
]

paths = ["01", "02", "03", "04", "05"]

all_data_rows = []

print("Iniciando a consolidação dos resultados...")

for dataset in datasets:
    for path in paths:
        result_file_path = os.path.join(base_dir, dataset, path, "result.csv")
        
        if not os.path.exists(result_file_path):
            print(f"[AVISO] Arquivo não encontrado e pulado: {result_file_path}")
            continue 
            
        try:
            df_temp = pd.read_csv(result_file_path, header=None)

            for index, row in df_temp.iterrows():
                all_data_rows.append({
                    "dataset": dataset,
                    "folder": path,
                    "shot": row[5], 
                    "acc": row[7]  
                })
        except Exception as e:
            print(f"Erro ao processar o arquivo {result_file_path}: {e}")

if all_data_rows:
    final_df = pd.DataFrame(all_data_rows)
    
    final_df["acc"] = pd.to_numeric(final_df["acc"], errors="coerce")
    final_df["shot"] = pd.to_numeric(final_df["shot"], errors="coerce")

    pivot_df = final_df.pivot_table(index=["dataset", "folder"], columns="shot", values="acc").reset_index()

    pivot_df.columns.name = None
    pivot_df = pivot_df.rename(columns={1: "acc_1shot", 5: "acc_5shot"})
    
    output_path = "result_geral_shots.csv"
    pivot_df.to_csv(output_path, index=False)
    
    print("\nConsolidação Concluída")
    print(f"Arquivo consolidado salvo em: {output_path}")
    print("\nAmostra do resultado final:")
    print(pivot_df.head())
else:
    print("\nNenhum dado foi processado. Verifique se os arquivos 'result.csv' existem nas subpastas.")