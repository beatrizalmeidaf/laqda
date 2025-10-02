#!/bin/bash
set -e 

cuda=2
commont=Bert-PN-addQ-CE-att
RAID_BASE_PATH="/raid/user_beatrizalmeida/laqda_results" 

for path in 01 02 03 04 05; do
    echo "Iniciando processamento para o path: $path"

    # --- Bloco do HuffPost ---
    HP_DIR="${RAID_BASE_PATH}/HuffPost/${path}"
    if [ -d "$HP_DIR" ]; then
        echo "Resultados para HuffPost path $path já existem. Pulando."
    else
        echo "Rodando HuffPost para o path $path..."
        python src_org/main.py --dataset HuffPost --dataFile data/HuffPost/few_shot/${path} --fileModelSave "$HP_DIR" --numKShot 5 --numDevice=$cuda --fileVocab="bert-base-uncased" --fileModelConfig="bert-base-uncased" --fileModel="bert-base-uncased" --sample 1 --numFreeze 6 --commont=$commont
        python src_org/main.py --dataset HuffPost --dataFile data/HuffPost/few_shot/${path} --fileModelSave "$HP_DIR" --numKShot 1 --numDevice=$cuda --fileVocab="bert-base-uncased" --fileModelConfig="bert-base-uncased" --fileModel="bert-base-uncased" --sample 1 --numFreeze 6 --commont=$commont
    fi

    # --- Bloco do 20News ---
    NEWS_DIR="${RAID_BASE_PATH}/20News/${path}"
    if [ -d "$NEWS_DIR" ]; then
        echo "Resultados para 20News path $path já existem. Pulando."
    else
        echo "Rodando 20News para o path $path..."
        python src_org/main.py --dataset 20News --dataFile data/20News/few_shot/${path} --fileModelSave "$NEWS_DIR" --numKShot 5 --numDevice=$cuda --fileVocab="bert-base-uncased" --fileModelConfig="bert-base-uncased" --fileModel="bert-base-uncased" --sample 1 --numFreeze 6 --commont=$commont
        python src_org/main.py --dataset 20News --dataFile data/20News/few_shot/${path} --fileModelSave "$NEWS_DIR" --numKShot 1 --numDevice=$cuda --fileVocab="bert-base-uncased" --fileModelConfig="bert-base-uncased" --fileModel="bert-base-uncased" --sample 1 --numFreeze 6 --commont=$commont
    fi

    # --- Bloco do Amazon ---
    AMAZON_DIR="${RAID_BASE_PATH}/Amazon/${path}"
    if [ -d "$AMAZON_DIR" ]; then
        echo "Resultados para Amazon path $path já existem. Pulando."
    else
        echo "Rodando Amazon para o path $path..."
        python src_org/main.py --dataset Amazon --dataFile data/Amazon/few_shot/${path} --fileModelSave "$AMAZON_DIR" --numKShot 5 --numDevice=$cuda --fileVocab="bert-base-uncased" --fileModelConfig="bert-base-uncased" --fileModel="bert-base-uncased" --sample 1 --numFreeze 6 --commont=$commont
        python src_org/main.py --dataset Amazon --dataFile data/Amazon/few_shot/${path} --fileModelSave "$AMAZON_DIR" --numKShot 1 --numDevice=$cuda --fileVocab="bert-base-uncased" --fileModelConfig="bert-base-uncased" --fileModel="bert-base-uncased" --sample 1 --numFreeze 6 --commont=$commont
    fi

    # --- Bloco do Reuters ---
    REUTERS_DIR="${RAID_BASE_PATH}/Reuters/${path}"
    if [ -d "$REUTERS_DIR" ]; then
        echo "Resultados para Reuters path $path já existem. Pulando."
    else
        echo "Rodando Reuters para o path $path..."
        python src_org/main.py --dataset Reuters --dataFile data/Reuters/few_shot/${path} --fileModelSave "$REUTERS_DIR" --numKShot 5 --numDevice=$cuda --fileVocab="bert-base-uncased" --fileModelConfig="bert-base-uncased" --fileModel="bert-base-uncased" --sample 1 --numFreeze 6 --commont=$commont --k 1 --numQShot 15
        python src_org/main.py --dataset Reuters --dataFile data/Reuters/few_shot/${path} --fileModelSave "$REUTERS_DIR" --numKShot 1 --numDevice=$cuda --fileVocab="bert-base-uncased" --fileModelConfig="bert-base-uncased" --fileModel="bert-base-uncased" --sample 1 --numFreeze 6 --commont=$commont --numQShot 15
    fi

done

echo "Script concluído"