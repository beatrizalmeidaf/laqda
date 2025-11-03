#!/bin/bash
set -e 

cuda=0
commont="exp_final_binary"
learning_rate=1.7e-5
RAID_BASE_PATH="/raid/user_beatrizalmeida/laqda_results_br_exp_binary"
DATASET_BASE_PATH="datasets-br/reviews_remapped_sentiment_binary"

DATASETS_TO_RUN=("B2WCorpus" "BrandsCorpus" "ReProCorpus" "UTLCorpus")

for dataset_name in "${DATASETS_TO_RUN[@]}"; do
    echo "============================================================"
    echo "PROCESSANDO DATASET: $dataset_name"
    echo "============================================================"

    for path in 01 02 03 04 05; do
        echo "Iniciando processamento para o fold: $path"

        RESULTS_DIR="${RAID_BASE_PATH}/${dataset_name}/${path}"
        SOURCE_DATA_DIR="${DATASET_BASE_PATH}/${dataset_name}/few_shot/${path}"
        RESULT_FILE="${RESULTS_DIR}/result.csv"

        # verifica se já está completo
        if [ -f "$RESULT_FILE" ] && grep -q "shot,1" "$RESULT_FILE" && grep -q "shot,5" "$RESULT_FILE"; then
            echo "Resultados COMPLETOS para $dataset_name no fold $path já existem. Pulando."
            continue 
        fi

        mkdir -p "$RESULTS_DIR"

        rm -f "$RESULT_FILE"

        # execução para 5-shot
        echo "Rodando ${dataset_name} (5-shot) para o fold $path..."
        python src_org/main.py \
            --dataset "$dataset_name" \
            --dataFile "$SOURCE_DATA_DIR" \
            --fileModelSave "$RESULTS_DIR" \
            --numKShot 5 \
            --numDevice="$cuda" \
            --fileVocab="neuralmind/bert-base-portuguese-cased" \
            --fileModelConfig="neuralmind/bert-base-portuguese-cased" \
            --fileModel="neuralmind/bert-base-portuguese-cased" \
            --sample 1 \
            --numFreeze 11 \
            --learning_rate ${learning_rate} \
            --commont="$commont"

        # execução para 1-shot
        echo "Rodando ${dataset_name} (1-shot) para o fold $path..."
        python src_org/main.py \
            --dataset "$dataset_name" \
            --dataFile "$SOURCE_DATA_DIR" \
            --fileModelSave "$RESULTS_DIR" \
            --numKShot 1 \
            --numDevice="$cuda" \
            --fileVocab="neuralmind/bert-base-portuguese-cased" \
            --fileModelConfig="neuralmind/bert-base-portuguese-cased" \
            --fileModel="neuralmind/bert-base-portuguese-cased" \
            --sample 1 \
            --numFreeze 11 \
            --learning_rate ${learning_rate} \
            --commont="$commont"
    done
done

echo "Processamento de todos os datasets concluído"
