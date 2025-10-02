#!/bin/bash
set -e 

cuda=1  
FreezeLayer=6
sample=100
commont=Bert-PN-addQ-CE-att
k=4
RAID_BASE_PATH="/raid/user_beatrizalmeida/laqda_results_intent" 

for path in 01 02 03 04 05; do
    echo "Iniciando processamento para o path: $path"

    # --- Bloco do Banking77 ---
    BANKING_DIR="${RAID_BASE_PATH}/Banking77/${path}"
    if [ -d "$BANKING_DIR" ]; then
        echo "Resultados para Banking77 path $path já existem. Pulando."
    else
        echo "Rodando Banking77 para o path $path..."
        python ./src_org/main.py --dataset Banking77 --dataFile data/BANKING77/few_shot/${path} --fileModelSave "$BANKING_DIR" --numKShot 1 --numQShot 5 --k=$k --sample=$sample --numDevice=$cuda --numFreeze=$FreezeLayer --commont=$commont --fileVocab="bert-base-uncased" --fileModelConfig="bert-base-uncased" --fileModel="bert-base-uncased"
        python ./src_org/main.py --dataset Banking77 --dataFile data/BANKING77/few_shot/${path} --fileModelSave "$BANKING_DIR" --numKShot 5 --numQShot 5 --k=$k --sample=$sample --numDevice=$cuda --numFreeze=$FreezeLayer --commont=$commont --fileVocab="bert-base-uncased" --fileModelConfig="bert-base-uncased" --fileModel="bert-base-uncased"
    fi

    # --- Bloco do Clinc150 ---
    CLINC_DIR="${RAID_BASE_PATH}/Clinc150/${path}"
    if [ -d "$CLINC_DIR" ]; then
        echo "Resultados para Clinc150 path $path já existem. Pulando."
    else
        echo "Rodando Clinc150 para o path $path..."
        python ./src_org/main.py --dataset Clinc150 --dataFile data/OOS/few_shot/${path} --fileModelSave "$CLINC_DIR" --numKShot 1 --numQShot 5 --k=$k --sample=$sample --numDevice=$cuda --numFreeze=$FreezeLayer --commont=$commont --fileVocab="bert-base-uncased" --fileModelConfig="bert-base-uncased" --fileModel="bert-base-uncased"
        python ./src_org/main.py --dataset Clinc150 --dataFile data/OOS/few_shot/${path} --fileModelSave "$CLINC_DIR" --numKShot 5 --numQShot 5 --k=$k --sample=$sample --numDevice=$cuda --numFreeze=$FreezeLayer --commont=$commont --fileVocab="bert-base-uncased" --fileModelConfig="bert-base-uncased" --fileModel="bert-base-uncased"
    fi

    # --- Bloco do Hwu64 ---
    HWU_DIR="${RAID_BASE_PATH}/Hwu64/${path}"
    if [ -d "$HWU_DIR" ]; then
        echo "Resultados para Hwu64 path $path já existem. Pulando."
    else
        echo "Rodando Hwu64 para o path $path..."
        python ./src_org/main.py --dataset Hwu64 --dataFile data/HWU64/few_shot/${path} --fileModelSave "$HWU_DIR" --numKShot 1 --numQShot 5 --k=$k --sample=$sample --numDevice=$cuda --numFreeze=$FreezeLayer --commont=$commont --fileVocab="bert-base-uncased" --fileModelConfig="bert-base-uncased" --fileModel="bert-base-uncased"
        python ./src_org/main.py --dataset Hwu64 --dataFile data/HWU64/few_shot/${path} --fileModelSave "$HWU_DIR" --numKShot 5 --numQShot 5 --k=$k --sample=$sample --numDevice=$cuda --numFreeze=$FreezeLayer --commont=$commont --fileVocab="bert-base-uncased" --fileModelConfig="bert-base-uncased" --fileModel="bert-base-uncased"
    fi
    
    # --- Bloco do Liu ---
    LIU_DIR="${RAID_BASE_PATH}/Liu/${path}"
    if [ -d "$LIU_DIR" ]; then
        echo "Resultados para Liu path $path já existem. Pulando."
    else
        echo "Rodando Liu para o path $path..."
        python ./src_org/main.py --dataset Liu --dataFile data/Liu/few_shot/${path} --fileModelSave "$LIU_DIR" --numKShot 1 --numQShot 5 --k=$k --sample=$sample --numDevice=$cuda --numFreeze=$FreezeLayer --commont=$commont --fileVocab="bert-base-uncased" --fileModelConfig="bert-base-uncased" --fileModel="bert-base-uncased"
        python ./src_org/main.py --dataset Liu --dataFile data/Liu/few_shot/${path} --fileModelSave "$LIU_DIR" --numKShot 5 --numQShot 5 --k=$k --sample=$sample --numDevice=$cuda --numFreeze=$FreezeLayer --commont=$commont --fileVocab="bert-base-uncased" --fileModelConfig="bert-base-uncased" --fileModel="bert-base-uncased"
    fi

done

echo "Script concluído"