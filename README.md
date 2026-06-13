# Trabalho 3 - Aplicações em Redes Neurais

Projeto de classificação de imagens utilizando o dataset **20 Fonts Classification**. O objetivo é identificar, a partir de uma imagem de uma palavra, qual fonte tipográfica foi usada para renderizá-la.

A entrega contempla análise exploratória de dados, pré-processamento, treinamento de redes neurais, avaliação com métricas adequadas, discussão dos resultados e organização dos artefatos finais.

## Equipe

- Amanda Evellin de Sousa Viana (2315774)
- João Victor da Silva Ferreira (2314387)
- Paulo Marconi Araújo Tomaz da Silva (2310435)
- Pedro Enrique Jordão Ramos Gama e Silva (2315773)
- Rogério Bruno de Almeida Júnior (2316922)

## Dataset

O dataset utilizado é o **20 Fonts Classification - A dataset for computer vision consisting of 20 common fonts**.

Fonte oficial: <https://www.kaggle.com/datasets/samoilovmikhail/20-fonts-classification>

O conjunto local fica em `dataset/20_fonts_classification/` e contém:

- `README.md`: descrição original do dataset.
- `metadata.csv`: metadados com `FileName`, `Font` e `Text`.
- `files/`: imagens PNG das palavras renderizadas.

Segundo a descrição do dataset, há 20 fontes, com 1000 palavras não repetidas para cada fonte. As imagens são visualmente preto e branco, possuem resolução de 200 x 50 pixels e contêm palavras em inglês com 3 a 8 letras.

Para baixar via Kaggle CLI:

```powershell
kaggle datasets download -d samoilovmikhail/20-fonts-classification -p dataset --unzip
```

Estrutura esperada após a extração:

```text
dataset/20_fonts_classification/
|-- README.md
|-- metadata.csv
`-- files/
```

## Estrutura do projeto

```text
.
|-- dataset/20_fonts_classification/
|-- notebooks/
|   |-- 01_analise_exploratoria_dados.ipynb
|   `-- 02_modelagem_treinamento_avaliacao.ipynb
|-- outputs/
|   |-- figures/
|   |-- models/
|   |-- splits/
|   |-- metrics.json
|   |-- classification_report_mlp.csv
|   |-- classification_report_cnn_batchnorm.csv
|   |-- classification_report_vgg16.csv
|   |-- confusion_matrix_mlp.csv
|   |-- confusion_matrix_cnn_batchnorm.csv
|   `-- confusion_matrix_vgg16.csv
|-- src/
|   |-- config.py
|   |-- data_utils.py
|   |-- evaluation.py
|   |-- font_classification.py
|   |-- models.py
|   |-- pipeline.py
|   |-- predict_font.py
|   |-- training.py
|   `-- visualization.py
`-- requirements.txt
```

## Notebook 1: Análise Exploratória de Dados

Arquivo: `notebooks/01_analise_exploratoria_dados.ipynb`

Este notebook é dedicado somente à compreensão do dataset antes da modelagem. Os principais tópicos abordados são:

- leitura do `README.md` e do `metadata.csv`;
- interpretação do problema como classificação supervisionada de imagens;
- verificação de valores ausentes, nomes duplicados e arquivos faltantes;
- contagem de imagens por fonte;
- análise do balanceamento das 20 classes;
- análise do tamanho das palavras renderizadas;
- verificação do formato, modo e resolução das imagens;
- visualização de amostras por classe;
- discussão das diferenças visuais entre fontes com serifa, sem serifa e fontes mais estilizadas;
- implicações da EDA para o pré-processamento.

Principais conclusões da EDA:

- O dataset possui 20.000 imagens e 20 classes.
- Cada fonte possui 1000 imagens, portanto o conjunto é balanceado.
- As imagens possuem resolução padronizada de 200 x 50 pixels.
- As palavras possuem de 3 a 8 letras.
- As imagens são visualmente preto e branco, mas estão salvas em modo RGB.
- Não foram identificados arquivos ausentes, nomes duplicados ou problemas estruturais relevantes.
- A principal dificuldade esperada é a semelhança visual entre algumas fontes, especialmente fontes sem serifa.

## Notebook 2: Modelagem, Treinamento e Avaliação

Arquivo: `notebooks/02_modelagem_treinamento_avaliacao.ipynb`

Este notebook executa e discute a etapa de modelagem. Os resultados são interpretados à medida que aparecem, de forma a conectar métricas, gráficos e decisões metodológicas.

Principais tópicos abordados:

- definição da estratégia de modelagem;
- configuração dos hiperparâmetros;
- pré-processamento das imagens;
- divisão estratificada em treino, validação e teste;
- treinamento de uma MLP Keras como baseline neural;
- treinamento de uma CNN Keras com BatchNormalization;
- treinamento de uma VGG16 pré-treinada com fine-tuning para comparação por transfer learning;
- análise das curvas de `loss` e `accuracy`;
- comparação entre MLP, CNN e VGG16 Fine-tuning no conjunto de teste;
- análise de `accuracy`, `macro F1` e `weighted F1`;
- métricas por classe com `precision`, `recall` e `F1-score`;
- identificação das melhores e piores classes por F1-score;
- interpretação da matriz de confusão;
- análise dos principais pares de fontes confundidos;
- discussão das limitações do modelo e melhorias futuras.

## Pré-processamento

O pipeline aplica os seguintes tratamentos:

- leitura dos metadados preservando textos como `"None"` sem convertê-los para valores ausentes;
- divisão estratificada em treino, validação e teste, na proporção 70/15/15;
- conversão das imagens para escala de cinza;
- redimensionamento para 128 x 32 pixels;
- normalização dos pixels para o intervalo `[0, 1]`;
- inversão da intensidade dos pixels, deixando o traço da letra como sinal alto e o fundo como sinal baixo;
- vetorização das imagens para entrada da MLP;
- organização das imagens como tensores `altura x largura x 1` para a CNN;
- repetição do canal em RGB para entrada da VGG16, preservando o contraste original esperado por modelos pré-treinados.

## Modelos utilizados

Foram comparados três modelos implementados com Keras:

- **MLP Keras**: baseline neural que recebe a imagem vetorizada. Serve para medir o desempenho de uma rede densa sem explorar explicitamente a estrutura espacial.
- **CNN Keras com BatchNormalization**: modelo convolucional principal, usando blocos `Conv2D`, `BatchNormalization` e `MaxPooling2D` para capturar padrões locais das letras.
- **VGG16 Fine-tuning**: modelo pré-treinado em ImageNet. Primeiro treina apenas a cabeça classificadora; depois descongela as últimas camadas para ajuste fino ao domínio das fontes.

| Modelo            | Tipo                 | Objetivo                           |
| ----------------- | -------------------- | ---------------------------------- |
| MLP Keras         | Rede densa           | Baseline neural                    |
| CNN BatchNorm     | CNN treinada do zero | Modelo principal                   |
| VGG16 Fine-tuning | Transfer learning    | Comparação com modelo pré-treinado |

A melhor versão de cada modelo é escolhida com base na acurácia de validação.

## Resultados obtidos

Após a execução do pipeline no dataset completo, o arquivo `outputs/metrics.json` consolidou as métricas de teste para os três modelos.

| Modelo            | Accuracy | Macro F1 | Weighted F1 |
| ----------------- | -------: | -------: | ----------: |
| MLP Keras         |   0.9693 |   0.9693 |      0.9693 |
| CNN BatchNorm     |   0.9930 |   0.9930 |      0.9930 |
| VGG16 Fine-tuning |   0.9287 |   0.9280 |      0.9280 |

| Modelo            | Tempo médio por época |
| ----------------- | --------------------: |
| MLP Keras         |                 0,8 s |
| CNN BatchNorm     |                  31 s |
| VGG16 Fine-tuning |                 126 s |

| Modelo            | Accuracy | Macro F1 | Tempo/época |
| ----------------- | -------: | -------: | ----------: |
| MLP Keras         |   96,93% |   96,93% |       0,8 s |
| CNN BatchNorm     |   99,30% |   99,30% |        31 s |
| VGG16 Fine-tuning |   92,87% |   92,80% |       126 s |

A MLP funciona como baseline neural forte, enquanto a CNN BatchNorm é o modelo principal por preservar a vizinhança espacial entre pixels e alcançou o melhor resultado geral. A VGG16 Fine-tuning entra como comparação externa com transfer learning e ajuste fino: ela avalia se representações pré-treinadas em ImageNet ajudam neste domínio, mas ficou abaixo dos modelos treinados diretamente nas imagens de fontes e teve custo computacional bem maior.

A VGG16 obteve melhor acurácia de validação de 0.9277, na época 23, após 8 épocas com a base congelada e 16 épocas de ajuste fino. O tempo médio ficou em cerca de 126 segundos por época, contra aproximadamente 31 segundos da CNN BatchNorm e 0,80 segundo da MLP Keras.

As principais confusões ocorreram entre fontes visualmente parecidas, especialmente fontes sem serifa. Esse comportamento é coerente com a natureza do problema, pois fontes como Arial, Helvetica, Calibri, Candara, SegoeUI e TrebuchetMS compartilham traços limpos, ausência de serifa e proporções semelhantes.

## Como executar

Crie e ative um ambiente Python 3.11 ou 3.12. Em seguida, instale as dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nos notebooks, selecione o interpretador `.venv\Scripts\python.exe` como kernel. Se ele não aparecer na lista do Jupyter, registre o kernel do projeto:

```powershell
python -m ipykernel install --user --name trabalho-redes-neurais --display-name "Python (.venv trabalho-redes-neurais)"
```

Execute o pipeline completo:

```powershell
python src/font_classification.py
```

Na primeira execução, a VGG16 com `--vgg-weights imagenet` pode baixar os pesos pré-treinados do Keras caso eles ainda não estejam no cache local. Se o download do host oficial estiver bloqueado, use `--vgg-weights none` para testar o fluxo sem pesos externos ou informe o caminho local para um arquivo `.h5` compatível com VGG16.

Esse comando realiza:

- leitura e validação do dataset;
- divisão estratificada dos dados;
- pré-processamento das imagens;
- treinamento da MLP Keras, da CNN BatchNorm e da VGG16 Fine-tuning;
- avaliação no conjunto de teste;
- geração de métricas, matriz de confusão e curvas;
- salvamento dos modelos Keras em `outputs/models/*.keras`.

Para alterar hiperparâmetros:

```powershell
python src/font_classification.py --image-width 128 --image-height 32 --mlp-epochs 24 --cnn-epochs 24 --vgg-head-epochs 8 --vgg-finetune-epochs 16
```

## Predição com modelo salvo

Após o treinamento, é possível classificar uma imagem individual:

```powershell
python src/predict_font.py dataset/20_fonts_classification/files/file_1.png
```

O script carrega por padrão `outputs/models/cnn_batchnorm_fonts_classifier.keras`, aplica o mesmo pré-processamento usado no treino e retorna as fontes mais prováveis.

Observação: o modelo `outputs/models/vgg16_fonts_classifier.keras` não é versionado porque tem cerca de 111 MB e ultrapassa o limite de 100 MB do GitHub. As métricas, metadados e demais artefatos da VGG16 permanecem em `outputs/`; o arquivo `.keras` pode ser regenerado executando o pipeline.

## Artefatos principais

- Notebook de análise exploratória: `notebooks/01_analise_exploratoria_dados.ipynb`
- Notebook de modelagem e avaliação: `notebooks/02_modelagem_treinamento_avaliacao.ipynb`
- Entrada do pipeline: `src/font_classification.py`
- Orquestração do pipeline: `src/pipeline.py`
- Leitura e pré-processamento: `src/data_utils.py`
- Modelos Keras: `src/models.py`
- Treinamento: `src/training.py`
- Métricas e matriz de confusão: `src/evaluation.py`
- Geração de figuras: `src/visualization.py`
- Script de predição: `src/predict_font.py`
- Métricas consolidadas: `outputs/metrics.json`
- Métricas por classe: `outputs/classification_report_*.csv`
- Matrizes de confusão: `outputs/confusion_matrix_*.csv`
- Figuras de análise: `outputs/figures/`
- Modelos treinados: `outputs/models/*.keras`, exceto `vgg16_fonts_classifier.keras`, omitido do Git por exceder o limite de tamanho do GitHub

## Limitações e melhorias futuras

A solução compara uma MLP com imagens vetorizadas, uma CNN que explora estrutura espacial e uma VGG16 pré-treinada com fine-tuning. Isso permite discutir tanto um baseline neural simples quanto arquiteturas externas de visão computacional.

Melhorias possíveis:

- testar diferentes quantidades de camadas convolucionais descongeladas na VGG16;
- aplicar aumento de dados controlado;
- ajustar hiperparâmetros com busca sistemática;
- avaliar desempenho em palavras novas fora do conjunto original;
- comparar diferentes resoluções de entrada;
- analisar erros por família tipográfica.
