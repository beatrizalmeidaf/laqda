#!/bin/bash
set -e 

cuda=0 
FreezeLayer=6
RAID_BASE_PATH="/raid/user_beatrizalmeida/qevasion" 
DATA_DIR="data/QEvasion_Task2" 
BASE_SAVE_DIR="${RAID_BASE_PATH}/QEvasion_Task2" 

echo "Rodando LAQDA na Tarefa de Evasão (Task 2) com múltiplos K-Shots"


for K in 35 45 75; do
    echo "Verificando/Rodando para k-shot = $K"
    
    # Salva cada K em sua própria subpasta 
    SAVE_DIR="${BASE_SAVE_DIR}/kshot_${K}" 
    RESULT_FILE="${SAVE_DIR}/result.csv"
    
    # Verifica se o result.csv final existe E se tem 1 linha (pois é 1 run por pasta)
    if [ -f "$RESULT_FILE" ] && [ $(wc -l < "$RESULT_FILE") -eq 1 ]; then
        echo "Resultados completos para K=$K já existem. Pulando."
    else
        echo "Rodando para k-shot = $K..."
        # Remove resultados parciais para garantir uma execução limpa
        rm -rf "$SAVE_DIR" 
        
        # Gera um 'commont' dinâmico para cada K
        commont="LAQDA-QEvasion-Task2-k${K}"

        python ./src_org/main.py \
            --dataset QEvasionTask2 \
            --dataFile "$DATA_DIR" \
            --fileModelSave "$SAVE_DIR" \
            --numKShot $K \
            --numQShot 15 \
            --numDevice=$cuda \
            --numFreeze=$FreezeLayer \
            --commont=$commont \
            --fileVocab="bert-base-uncased" \
            --fileModelConfig="bert-base-uncased" \
            --fileModel="bert-base-uncased"
    fi
done

echo "Script concluído"