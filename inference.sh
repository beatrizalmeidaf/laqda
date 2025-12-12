#!/bin/bash

#SBATCH --job-name=laqda_preds         # Nome do strabalho
#SBATCH --output=laqda_preds_saida_%j.log   # Arquivo para onde a saída padrão vai
#SBATCH --error=laqda_preds_erro_%j.log     # Arquivo para onde os erros vão
#SBATCH --time=08:00:00              # Tempo máximo de execução (8 horas)
#SBATCH --partition=h100n3           # Partição 
#SBATCH --gres=gpu:h100:1            # Pede UMA GPU

source /home/user_beatrizalmeida/laqda_venv/bin/activate

cd /home/user_beatrizalmeida/laqda/

echo "Iniciando geração de predições..."

python inference.py

echo "Concluído."