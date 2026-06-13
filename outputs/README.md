# Artefatos finais do experimento

Esta pasta contém os resultados finais gerados pelo pipeline descrito no README principal e nos notebooks. Os modelos avaliados na entrega são:

| Modelo            | Papel no experimento                                |
| ----------------- | --------------------------------------------------- |
| MLP Keras         | Baseline neural com imagem vetorizada               |
| CNN BatchNorm     | Modelo principal treinado do zero                   |
| VGG16 Fine-tuning | Comparação por transfer learning com pesos ImageNet |

## Arquivos principais

- `metrics.json`: métricas consolidadas, histórico de treinamento, configuração usada e principais confusões.
- `classification_report_*.csv`: precision, recall, F1-score e suporte por classe.
- `confusion_matrix_*.csv`: matrizes de confusão por modelo.
- `splits/`: arquivos CSV com a divisão estratificada de treino, validação e teste.
- `figures/`: figuras usadas no README e nos notebooks.
- `models/`: modelos Keras finais e metadados necessários para predição. O arquivo `vgg16_fonts_classifier.keras` foi omitido do repositório porque tem cerca de 111 MB e ultrapassa o limite de 100 MB do GitHub; ele pode ser regenerado executando o pipeline.

## Resultados finais

| Modelo            | Accuracy | Macro F1 | Tempo médio por época |
| ----------------- | -------: | -------: | --------------------: |
| MLP Keras         |   0.9693 |   0.9693 |                 0,8 s |
| CNN BatchNorm     |   0.9930 |   0.9930 |                  31 s |
| VGG16 Fine-tuning |   0.9287 |   0.9280 |                 126 s |
