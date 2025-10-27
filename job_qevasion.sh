#!/bin/bash

#SBATCH --job-name=laqda_qevasion      # Nome do trabalho
#SBATCH --output=qevasion_saida_%j.log # Arquivo de saída
#SBATCH --error=qevasion_erro_%j.log   # Arquivo de erro
#SBATCH --time=28:00:00                # Tempo máximo de execução (28 horas)
#SBATCH --partition=h100n3             # Partição 
#SBATCH --gres=gpu:h100:1              # Pede UMA GPU 


source /home/user_beatrizalmeida/laqda_venv/bin/activate

echo "=========================================================="
echo "Data de início: $(date)"
echo "Nó de execução: $(hostname)"
echo "GPUs alocadas: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

cd /home/user_beatrizalmeida/laqda/

bash run_qevasion.sh

echo "=========================================================="
echo "Job LAQDA QEvasiON Concluído!"
echo "Data de término: $(date)"
echo "=========================================================="