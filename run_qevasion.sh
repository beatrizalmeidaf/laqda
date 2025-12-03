#!/bin/bash
set -e 

cuda=0 
FreezeLayer=6
RAID_BASE_PATH="/raid/user_beatrizalmeida/qevasion" 
BASE_DATA_DIR="data/QEvasionCorpus/few_shot" 
BASE_SAVE_DIR="${RAID_BASE_PATH}/QEvasion_Augmented_Results" 

echo "Rodando LAQDA (com BGE-M3) na Tarefa de Evasão (5-Fold)"

MODEL_NAME="BAAI/bge-m3"
K_SHOT=55 # O k-shot que você quer testar

# --- Loop pelos 5 Folds ---
for fold in 01 02 03 04 05; do
    echo "--- Verificando/Rodando Fold $fold (k-shot = $K_SHOT) ---"
    
    # --- CORREÇÃO IMPORTANTE ---
    # Define o DATA_DIR para o caminho completo do fold atual
    DATA_DIR="${BASE_DATA_DIR}/${fold}" # Ex: "data/QEvasionCorpus/few_shot/01"
    
    # Salva cada fold em uma subpasta de resultados
    SAVE_DIR="${BASE_SAVE_DIR}/kshot_${K_SHOT}_fold_${fold}" 
    RESULT_FILE="${SAVE_DIR}/result.csv"
    
    if [ -f "$RESULT_FILE" ] && [ $(wc -l < "$RESULT_FILE") -eq 1 ]; then
        echo "Resultados completos para Fold $fold já existem. Pulando."
    else
        echo "Rodando Fold $fold..."
        rm -rf "$SAVE_DIR" 
        
        commont="LAQDA-BGE-M3-Fold${fold}-k${K_SHOT}"

        # O main.py agora recebe o caminho completo, ex: data/QEvasionCorpus/few_shot/01
        python src/main.py \
            --dataset "QEvasionTask2" \
            --dataFile "$DATA_DIR" \
            --fileModelSave "$SAVE_DIR" \
            --numKShot $K_SHOT \
            --numQShot 15 \
            --numDevice=$cuda \
            --numFreeze=$FreezeLayer \
            --commont=$commont \
            --fileVocab="$MODEL_NAME" \
            --fileModelConfig="$MODEL_NAME" \
            --fileModel="$MODEL_NAME"
    fi
done

echo "Script de 5-Folds concluído"