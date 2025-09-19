## Estrutura dos Arquivos

### `data_loader.py`

Responsável por carregar e organizar o dataset em **episódios de treinamento**, em vez de usar o carregamento tradicional e aleatório.

Cada episódio é composto por:

* **Conjunto de Suporte (Support Set):** pequeno número de exemplos rotulados que “ensinam” o modelo sobre novas classes.
* **Conjunto de Consulta (Query Set):** exemplos que o modelo deve classificar com base no que aprendeu do conjunto de suporte.


### `encoder.py`

Define o modelo de classificação de texto.

Fluxo principal:

1. **Codificação de Texto:** usa um modelo BERT pré-treinado para converter sentenças em vetores numéricos (embeddings).
2. **Protótipos Iniciais:** calcula um vetor médio para cada classe a partir do conjunto de suporte.
3. **Refinamento de Protótipos:** aplica um módulo *Sampler* para selecionar exemplos relevantes do conjunto de consulta, gerando protótipos mais robustos.
4. **Classificação:** compara embeddings do conjunto de consulta com os protótipos refinados.

> O método é **transdutivo**, pois aproveita informações do conjunto de consulta para melhorar a performance dentro do próprio episódio.


### `losses.py`

Define a classe `Loss_fn`, que combina cálculo de perda e métricas de desempenho.

* **Perda (Loss):**

  * **Principal:** mede a qualidade da classificação usando os protótipos refinados.
  * **Auxiliar:** atua como regularização, garantindo que os exemplos escolhidos pelo *Sampler* mantenham consistência com os protótipos originais.

* **Métricas de Performance:** precisão, F1-score, acurácia, entre outras, para avaliar o desempenho do modelo de forma interpretável.


### `main.py`

Arquivo principal de execução do projeto.

Etapas principais:

1. **Configuração:** lê parâmetros da linha de comando (taxa de aprendizado, épocas, caminhos, etc.), define sementes aleatórias e configura o ambiente (CPU/GPU).
2. **Inicialização:** instancia

   * *DataLoaders* para treino, validação e teste,
   * o modelo (`MyModel`),
   * o otimizador (AdamW),
   * o *scheduler* de taxa de aprendizado.
3. **Treinamento:** executa o loop de treino, calcula perdas, atualiza parâmetros e valida o desempenho.
4. **Avaliação:** realiza o teste final no conjunto de avaliação.


### `utils.py`

Fornece funções utilitárias, incluindo a configuração de argumentos que serão repassados à aplicação.


