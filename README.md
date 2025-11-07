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
> Após o download, atualize o caminho para o modelo dentro do arquivo `run.sh` de acordo com o diretório local onde ele foi salvo.
> Os parâmetros específicos para cada conjunto de dados utilizados no artigo estão definidos no `run.sh`.

```bash
sh run.sh
```

### Resultados Datasets Classificação

A performance do modelo LAQDA foi avaliada em quatro datasets de classificação de texto amplamente utilizados. A tabela abaixo resume a acurácia média e o desvio padrão (%) calculados a partir de 5 execuções independentes para cada dataset, nos cenários de 1-shot e 5-shot.

| Dataset    | Acurácia (1-shot) | Acurácia (5-shot) |
| :--------- | :---------------- | :---------------- |
| **Amazon** | 79.65% ± 6.88%    | 88.31% ± 3.63%    |
| **HuffPost** | 56.65% ± 1.62%    | 72.34% ± 1.94%    |
| **Reuters** | 92.55% ± 1.25%    | 95.34% ± 1.15%    |
| **20News** | 77.12% ± 4.31%    | 85.37% ± 2.91%    |

<p align="center">
  <img src="results/resultado-1shot.png" alt="Resultados 1-shot" width="80%"/>
  <img src="results/resultado-5shot.png" alt="Resultados 5-shot" width="80%"/>
</p>
<p align="center"><em>Figura 1 – Resultados por pasta nos datasets de classificação de texto</em></p>


#### Análise dos Resultados

Os resultados demonstram a eficácia da abordagem LAQDA em melhorar a performance de modelos de meta-learning, especialmente em cenários de extrema escassez de dados (1-shot).

  * **Alto Desempenho em Cenários de 1-shot:** O desempenho notável, especialmente em datasets como **Reuters (92.55%)** e **Amazon (79.65%)**, valida a principal inovação do LAQDA: o **Query-Data-Augmenter (QDA)**. Ao usar os próprios exemplos do conjunto de consulta para refinar um protótipo criado a partir de um único exemplo, o modelo consegue estabilizar e corrigir a representação da classe, mitigando o risco de um único exemplo de suporte ser um outlier e não representativo.

  * **Impacto da Semântica dos Rótulos:** O **Label-Adapter (LA)** contribui para criar um espaço de características mais coeso, onde as classes são mais distinguíveis. Isso é evidenciado pela performance robusta em datasets com classes bem definidas, como o Reuters, onde o desvio padrão baixo indica uma grande estabilidade nos resultados.

  * **Consistência e Generalização:** O modelo apresenta um ganho de performance consistente ao passar de 1-shot para 5-shot em todos os cenários. Isso mostra que, embora seja otimizado para poucos dados, sua capacidade de criar protótipos de qualidade escala bem com o aumento (ainda que pequeno) de informações disponíveis. A variação de performance entre os datasets (ex: HuffPost vs. Reuters) reflete a complexidade intrínseca e a separabilidade das classes em cada um, mas a metodologia LAQDA se prova eficaz em todos eles.

<p align="center">
  <img src="results/media-1shot.png" alt="Média 1-shot" width="80%"/>
  <img src="results/media-5shot.png" alt="Média 5-shot" width="80%"/>
</p>
<p align="center"><em>Figura 2 – Acurácia média por dataset</em></p>


### Resultados em Datasets de Intenção

A performance do modelo LAQDA também foi avaliada em quatro datasets de classificação de intenção amplamente utilizados. A tabela abaixo resume a acurácia média e o desvio padrão (%) calculados a partir de 5 execuções independentes para cada dataset, nos cenários de 1-shot e 5-shot.

| Dataset       | Acurácia (1-shot) | Acurácia (5-shot) |
| :------------ | :---------------- | :---------------- |
| **Banking77** | 89.18% ± 1.19%    | 94.77% ± 0.52%    |
| **Clinc150**  | 96.98% ± 0.53%    | 98.58% ± 0.20%    |
| **Hwu64**     | 85.22% ± 1.47%    | 91.76% ± 1.33%    |
| **Liu**       | 87.96% ± 1.88%    | 93.50% ± 1.13%    |

<p align="center">
  <img src="results/resultado_intent_1shot.png" alt="Resultados Intenção 1-shot" width="80%"/>
  <img src="results/resultado_intent_5shot.png" alt="Resultados Intenção 5-shot" width="80%"/>
</p>
<p align="center"><em>Figura 3 – Resultados por pasta nos datasets de intenção</em></p>


#### Análise dos Resultados

Os resultados confirmam a robustez do LAQDA em cenários de classificação de intenção:

* **Excelente Desempenho Geral:** Datasets como **Clinc150** atingem quase **97% de acurácia em 1-shot** e superam **98% em 5-shot**, evidenciando que a abordagem lida bem mesmo em tarefas com grande número de intenções distintas.
* **Estabilidade em Dados Complexos:** O **Banking77**, conhecido por possuir intenções semanticamente próximas, manteve acurácias consistentes com **desvio padrão baixo**, o que indica a capacidade do modelo em diferenciar nuances de intenções similares.
* **Impacto do Aumento de Suporte:** Assim como nos datasets de classificação de texto, há **ganhos consistentes ao passar de 1-shot para 5-shot**, com destaque para o Hwu64, que apresentou salto de mais de 6 pontos percentuais, demonstrando que o QDA e o Label-Adapter reforçam representações mesmo em domínios mais desafiadores.

<p align="center">
  <img src="results/media_intent_1shot.png" alt="Média Intenção 1-shot" width="80%"/>
  <img src="results/media_intent_5shot.png" alt="Média Intenção 5-shot" width="80%"/>
</p>
<p align="center"><em>Figura 4 – Acurácia média por dataset de intenção</em></p>  


#### Resultados em Datasets PT-BR

A abordagem **LAQDA** também foi avaliada em um conjunto de **datasets brasileiros de reviews e análises de sentimentos**, a fim de verificar sua capacidade de generalização em diferentes idiomas e domínios textuais.
A figura abaixo apresenta a comparação entre os cenários de **1-shot** e **5-shot**.

<p align="center">
  <img src="results/media-br-reviews.png" alt="Comparação de Desempenho BR" width="80%"/>
</p>
<p align="center"><em>Figura 5 – Comparação de desempenho (1-shot vs 5-shot) em datasets PT-BR</em></p>

#### Análise dos Resultados (PT-BR)

Os resultados mostram uma tendência clara de melhoria com o aumento do número de exemplos por classe (de 1-shot para 5-shot), validando a robustez do LAQDA em contextos de língua portuguesa.

* **Consistência entre domínios:** O modelo mantém altas acurácias médias na maioria dos datasets brasileiros, demonstrando que as estratégias de Query-Data-Augmenter (QDA) e Label-Adapter (LA) preservam a qualidade das representações mesmo em textos com gírias e ruídos típicos de avaliações online.

* **Ganho expressivo em cenários desafiadores:** Em datasets mais heterogêneos, como UTLCorpus, há ganhos superiores a 6 pontos percentuais, refletindo a capacidade do LAQDA em ajustar representações quando há aumento do suporte supervisionado.

* **Conclusão geral:** O comportamento observado confirma que o LAQDA não apenas generaliza entre idiomas, mas também mantém sua eficiência em ambientes de linguagem natural informal, típicos do português brasileiro.

