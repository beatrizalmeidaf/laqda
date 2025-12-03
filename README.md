# LAQDA – Aprimorando Meta-Learning para Classificação de Texto Few-Shot

Esse repositório contém o código do **LAQDA**, uma abordagem voltada para melhorar o meta-learning em tarefas de classificação de texto com poucos exemplos (Few-Shot Learning).


## Guia Rápido de Uso

### 1. Criar o Ambiente

```bash
conda create -n LAQDA 
conda activate LAQDA
pip install -r requirements.txt
```

### 2. Executar o Modelo

> **Atenção:** antes de rodar o projeto, é necessário baixar o modelo **bert-base-uncased** disponível em:
> [https://huggingface.co/google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)
>
> Após o download, atualize o caminho para o modelo dentro do arquivo `run_qevasion.sh` de acordo com o diretório local onde ele foi salvo.


```bash
sbatch job_qevasion.sh
```
