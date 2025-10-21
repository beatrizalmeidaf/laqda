import json
import sys

def converter(caminho_entrada, caminho_saida):
    """
    Converte um arquivo JSON (contendo um array de objetos) para o formato
    JSON Lines (JSONL), ajustando as chaves para o formato esperado pelo script de treino.
    """
    print(f"Convertendo '{caminho_entrada}' para '{caminho_saida}'")
    try:
        with open(caminho_entrada, 'r', encoding='utf-8') as f_in:
            dados = json.load(f_in)

        with open(caminho_saida, 'w', encoding='utf-8') as f_out:
            for item in dados:
                novo_item = {
                    "sentence": item.get("text"),  
                    "label": item.get("label")     
                }
                f_out.write(json.dumps(novo_item, ensure_ascii=False) + '\n')
                
    except Exception as e:
        print(f"Ocorreu um erro ao converter o arquivo: {e}")
        sys.exit(1) 

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Uso: python converter_formato.py <arquivo_de_entrada.json> <arquivo_de_saida.jsonl>")
        sys.exit(1)

    caminho_entrada = sys.argv[1]
    caminho_saida = sys.argv[2]
    converter(caminho_entrada, caminho_saida)