# Projeto Data Splitter

## Introdução e motivação

Para diversos tipos de tarefas envolvendo treinamentos de visão computacional, é necessário realizar uma espécie de downscailing da dificuldade dos datasets, para ir treinando o modelo em passos. Uma metodologia útil para tal é o treinamento da rede em passos, ou seja, fornecer imagens cujas bounding boxes sejam relativamente maiores, de modo que a rede possa aprender primeiro a reconhecer o objeto em contextos simples, e futuramente fornecer dados mais complexos.

## Metodologia

O algoritmo do Data Splitter analisa o banco de dados de imagens fornecido, separa as imagens cujas bounding boxes ocupem pelo menos x% da área da imagem e realiza o particionamento em train, test, val de maneira pseudoaleatória utilizando o algoritmo Stratified Group Shuffle Split, mantendo uma proporção de 60% para treino, 20% para teste e 20% para validação.

## Parâmetros

O algoritmo funciona com base nos seguintes parâmetros:

```$python3 data_splitter.py <caminho_para_dataset> <nome_do_yaml> -p PERCENTUAL -i -d DESTINATION```

Onde:

- ```<caminho_para_dataset>``` é o caminho para a raiz do diretório onde estão os dados
- ```<nome_do_yaml>``` é o nome do arquivo .yaml contendo informações acerca do dataset. O valor default definido é ```data.yaml```.
- ```-p PERCENTUAL, --percent PERCENTUAL``` é o valor da porcentagem da imagem que uma bounding box deve ocupar para ser incluída no novo dataset. Nota-se que o valor é inclusivo, ou seja, todas as imagens que contiverem uma Bounding Box que ocupe espaço >= a p% da imagem serão consideradas no novo dataset.
- ```-i, --invert``` Alguns datasets invertem a lógica padrão de diretórios, que seguiria a seguinte árvore:

    ```raw
    .
    ├── test
    │   ├── images
    │   └── labels
    ├── train
    │   ├── images
    │   └── labels
    └── val
        ├── images
        └── labels
    ```

    Estes projetos, por sua vez, usam a seguinte distribuição:

    ```raw
    .
    ├── images
    │   ├── test
    │   ├── train
    │   └── val
    └── labels
        ├── test
        ├── train
        └── val
    ```

    Neste caso, a flag ```-i``` indica que a ordem dos diretórios foi trocada, de modo que o programa percorra a árvore de arquivos corretamente.
- ```-d DESTINATION --destination DESTINATION``` é o **NOME** do novo dataset que será gerado ao lado do dataset antigo. O dataset novo obedecerá à estrutura:

    ```raw
        <DESTINATION>
        |---data.yaml
        |
        ├── test
        │   ├── images
        │   └── labels
        ├── train
        │   ├── images
        │   └── labels
        └── val
            ├── images
            └── labels

    ```
