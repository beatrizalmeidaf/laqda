import pandas as pd
import os

base_dirs = {
    "categ": "/raid/user_beatrizalmeida/laqda_categ_results_br",
    "hate": "/raid/user_beatrizalmeida/laqda_hate_results_br",
    "intent": "/raid/user_beatrizalmeida/laqda_intent_results_br",
    "review": "/raid/user_beatrizalmeida/laqda_review_results_br",
}

all_results = []


for dtype, path in base_dirs.items():
    file_path = os.path.join(path, "result_geral_shots.csv")
    if not os.path.exists(file_path):
        print(f"Arquivo não encontrado: {file_path}")
        continue

    df = pd.read_csv(file_path)

    # calcula média por dataset
    df_mean = df.groupby("dataset", as_index=False)[["acc_1shot", "acc_5shot"]].mean()

    # adiciona a coluna com o tipo
    df_mean["type"] = dtype

    all_results.append(df_mean)

if all_results:
    df_final = pd.concat(all_results, ignore_index=True)
    df_final = df_final[["type", "dataset", "acc_1shot", "acc_5shot"]]
    df_final.to_csv("/raid/user_beatrizalmeida/laqda_result_br.csv", index=False)
    print("Arquivo final salvo em /raid/user_beatrizalmeida/laqda_result_br.csv")
else:
    print("Nenhum resultado encontrado.")
