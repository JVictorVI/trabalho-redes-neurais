# 20 Fonts Classification

## Descrição do dataset

Este dataset foi utilizado no Trabalho 3 de Aplicações em Redes Neurais para resolver um problema de classificação de imagens. A tarefa consiste em receber uma imagem contendo uma palavra renderizada e prever qual fonte tipográfica foi usada.

O problema é supervisionado, pois cada imagem possui um rótulo conhecido na coluna `Font` do arquivo `metadata.csv`. Assim, os modelos podem aprender a relação entre os pixels da imagem e a fonte correspondente.

## Estrutura dos arquivos

```text
20_fonts_classification/
|-- README.md
|-- metadata.csv
`-- files/
```

- `README.md`: descrição do dataset e explicações adicionais para o projeto.
- `metadata.csv`: arquivo tabular com os metadados das imagens.
- `files/`: pasta com as imagens PNG.

## Metadados

O arquivo `metadata.csv` possui três colunas principais:

- `FileName`: nome do arquivo de imagem dentro da pasta `files/`.
- `Font`: classe da imagem, ou seja, a fonte usada para renderizar a palavra.
- `Text`: palavra exibida na imagem.

Exemplo:

```text
FileName,Font,Text
file_1.png,CourierNew,Sediment
file_2.png,Arial,Guest
file_3.png,CenturyGothic,Socket
```

## Classes

O dataset possui 20 classes, uma para cada fonte:

- Arial
- TimesNewRoman
- Verdana
- Georgia
- CourierNew
- TrebuchetMS
- Palatino
- Garamond
- Bookman
- ComicSansMS
- Lobster
- LucidaSans
- MS_Sans_Serif
- Helvetica
- MonotypeCorsiva
- CenturyGothic
- FranclinGothic
- Calibri
- Candara
- SegoeUI

Cada classe possui 1000 imagens, totalizando 20.000 imagens. Como todas as classes possuem a mesma quantidade de exemplos, o dataset é balanceado.

## Características das imagens

- Formato: PNG.
- Resolução original: 200 x 50 pixels.
- Aparência: imagens visualmente preto e branco.
- Modo de leitura observado: RGB.
- Conteúdo: palavras em inglês.
- Tamanho das palavras: de 3 a 8 letras.

Mesmo que as imagens sejam visualmente preto e branco, elas podem ser lidas como RGB por bibliotecas como PIL. Por isso, no pré-processamento do projeto, as imagens são convertidas para escala de cinza.

## Uso no projeto

No pipeline desenvolvido, o dataset é usado da seguinte forma:

1. O arquivo `metadata.csv` é lido.
2. Cada imagem é associada à sua classe pela coluna `Font`.
3. Os dados são divididos de forma estratificada em treino, validação e teste.
4. As imagens são convertidas para escala de cinza.
5. As imagens são redimensionadas para 128 x 32 pixels.
6. Os pixels são normalizados para o intervalo `[0, 1]`.
7. A intensidade é invertida para que o traço da letra fique com valor alto e o fundo fique com valor baixo.
8. As imagens são vetorizadas e usadas como entrada nos modelos.

## Observações importantes

Algumas linhas do `metadata.csv` possuem a palavra literal `None` na coluna `Text`. Essa palavra faz parte do dataset e deve ser tratada como texto normal, não como valor ausente.

Por isso, ao ler o CSV com pandas, o projeto usa:

```python
pd.read_csv("metadata.csv", keep_default_na=False)
```

Esse cuidado impede que a palavra `"None"` seja convertida automaticamente para `NaN`.

## Fonte oficial

Dataset Kaggle: <https://www.kaggle.com/datasets/samoilovmikhail/20-fonts-classification>
