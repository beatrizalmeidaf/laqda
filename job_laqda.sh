#!/bin/bash

#SBATCH --job-name=laqda_br_run      # Nome do trabalho (mudei)
#SBATCH --output=laqda_saida_%j.log  # Arquivo de saída (mudei)
#SBATCH --error=laqda_erro_%j.log    # Arquivo de erro (mudei)
#SBATCH --time=08:00:00              # Tempo máximo de execução (8 horas)
#SBATCH --partition=h100n3       
#SBATCH --gres=gpu:h100:1            # Pede UMA GPU

# ativa o ambiente virtual do LAQDA
source /home/user_beatrizalmeida/laqda_venv/bin/activate

echo "=========================================================="
echo "Data de início: $(date)"
echo "Nó de execução: $(hostname)"
echo "GPUs alocadas: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="


cd /home/user_beatrizalmeida/laqda/

bash run_br.sh

echo "=========================================================="
echo "Job LAQDA Concluído!"
echo "=========================================================="