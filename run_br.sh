#!/bin/bash
set -e 

cuda=0
commont="exp1_freeze10_lr5e-5"
learning_rate=5e-5
RAID_BASE_PATH="/raid/user_beatrizalmeida/laqda_results_br_exp1"
DATASET_BASE_PATH="datasets-br/reviews" 
TEMP_DATA_DIR="temp_data_for_training"

DATASETS_TO_RUN=("B2WCorpus" "BrandsCorpus" "ReProCorpus" "UTLCorpus")

for dataset_name in "${DATASETS_TO_RUN[@]}"; do
    echo "============================================================"
    echo "PROCESSANDO DATASET: $dataset_name"
    echo "============================================================"

    for path in 01 02 03 04 05; do
        echo "Iniciando processamento para o fold: $path"

        RESULTS_DIR="${RAID_BASE_PATH}/${dataset_name}/${path}"
        SOURCE_DATA_DIR="${DATASET_BASE_PATH}/${dataset_name}/few_shot/${path}"
        TEMP_FOLD_DIR="${TEMP_DATA_DIR}/${dataset_name}/${path}"

        # verificação de conteúdo do arquivo de resultado 
        RESULT_FILE="${RESULTS_DIR}/result.csv"

        # verifica 3 coisas:
        # 1. O arquivo result.csv existe? (-f)
        # 2. contém a linha do 1-shot? (grep -q "shot,1")
        # 3. contém a linha do 5-shot? (grep -q "shot,5")
        if [ -f "$RESULT_FILE" ] && grep -q "shot,1" "$RESULT_FILE" && grep -q "shot,5" "$RESULT_FILE"; then
            echo "Resultados COMPLETOS para $dataset_name no fold $path já existem. Pulando."
            continue 
        fi
        
        # garante que as pastas de trabalho existam
        mkdir -p "$RESULTS_DIR"
        mkdir -p "$TEMP_FOLD_DIR"

        echo "Preparando dados temporários em: $TEMP_FOLD_DIR"
        python converter_formato.py "${SOURCE_DATA_DIR}/train.json" "${TEMP_FOLD_DIR}/train.json"
        python converter_formato.py "${SOURCE_DATA_DIR}/valid.json" "${TEMP_FOLD_DIR}/valid.json"
        python converter_formato.py "${SOURCE_DATA_DIR}/test.json" "${TEMP_FOLD_DIR}/test.json"
        
        rm -f "$RESULT_FILE"
        
        # execução para 5-shot
        echo "Rodando ${dataset_name} (5-shot) para o fold $path..."
        python src_org/main.py \
            --dataset "$dataset_name" \
            --dataFile "$TEMP_FOLD_DIR" \
            --fileModelSave "$RESULTS_DIR" \
            --numKShot 5 \
            --numDevice="$cuda" \
            --fileVocab="neuralmind/bert-base-portuguese-cased" \
            --fileModelConfig="neuralmind/bert-base-portuguese-cased" \
            --fileModel="neuralmind/bert-base-portuguese-cased" \
            --sample 1 \
            --numFreeze 10 \
            --learning_rate ${learning_rate} \
            --commont="$commont"

        # execução para 1-shot
        echo "Rodando ${dataset_name} (1-shot) para o fold $path..."
        python src_org/main.py \
            --dataset "$dataset_name" \
            --dataFile "$TEMP_FOLD_DIR" \
            --fileModelSave "$RESULTS_DIR" \
            --numKShot 1 \
            --numDevice="$cuda" \
            --fileVocab="neuralmind/bert-base-portuguese-cased" \
            --fileModelConfig="neuralmind/bert-base-portuguese-cased" \
            --fileModel="neuralmind/bert-base-portuguese-cased" \
            --sample 1 \
            --numFreeze 10 \
            --learning_rate ${learning_rate} \
            --commont="$commont"
    done
done

echo "Limpando dados temporários"
rm -rf "$TEMP_DATA_DIR"

echo "Processamento de todos os datasets concluído"