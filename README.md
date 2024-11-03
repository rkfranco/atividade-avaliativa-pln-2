# Projeto de Análise de Texto

## Descrição
Este projeto tem como objetivo analisar um corpus de texto utilizando diferentes técnicas de vetorização e análise de dados. As técnicas incluem Bag of Words (BoW), TF-IDF e word embeddings. Os resultados são interpretados e visualizados para entender melhor os dados.
- **OBS**: Foi criado um arquivo especifico para cada secção deste trabalho. O nome do arquivo é o mesmo que o da secção.

## Dimensões e Critérios de Avaliação

### [Bag of Words](bag_of_words.py)
- **Critérios de Avaliação**: Os dados foram vetorizados com BoW e foram calculados termos mais frequentes do corpus, incluindo análises com visualizações?
- **Peso**: 0,1

### [TF-IDF](tf_idf.py)
- **Critérios de Avaliação**:
  - Os dados foram vetorizados com TF-IDF e foram calculadas as similaridades entre os documentos?
  - Foram analisados clusters, sendo visualizados e interpretados com PCA ou outra técnica semelhante?
- **Peso**: 0,2

### Bônus (TODO)
- **Critérios de Avaliação**: Foram empregados n-gramas?
- **Peso**: 0,1

### [Embeddings](embeddings.py)
- **Critérios de Avaliação**: Foram aplicados word embeddings e técnicas relacionadas (similaridade, sistemas de recomendação ou busca, análise de tópicos, identificação de entidades)?
- **Peso**: 0,3

### Interpretação dos Resultados (TODO)
- **Critérios de Avaliação**: Produção de texto analisando os resultados obtidos e a adequação das diferentes técnicas e algoritmos à base de dados escolhida.
- **Peso**: 0,4

## Instalação
Instruções sobre como instalar e configurar o projeto.

```bash
# Clone o repositório
git clone https://github.com/rkfranco/atividade-avaliativa-pln-2.git

# Navegue até o diretório do projeto
cd atividade-avaliativa-pln-2

# Instale as dependências
pip install -r requirements.txt
