"""
Embeddings com spaCy e SBERT

Alunos: Rodrigo K. Franco, Maria Eduarda Krutzsch, Luan L. Guarnieri, Nicole B., Gustavo Baroni, Ana Carolina

Critérios de Avaliação:
Foram aplicados word embeddings e técnicas relacionadas (similaridade, sistemas de recomendação ou busca, análise de tópicos, identificação de entidades)? (Peso: 0,3)

Descrição das Implementações:
== Embeddings com spaCy e Cálculo de Similaridade ==
  - Utilizamos o modelo de embeddings 'pt_core_news_lg' do spaCy para representar semanticamente os textos. A similaridade entre os documentos foi calculada usando a 'similaridade de cosseno', que avalia a proximidade entre dois vetores em termos do ângulo entre eles.
  - A matriz de similaridade foi visualizada em um 'heatmap', em que cada célula mostra a similaridade entre dois documentos. A diagonal principal apresenta sempre o valor 1, indicando que cada documento é 100% similar a si mesmo.

== Sistema de Recomendação e Busca Semântica ==
  - Implementamos um sistema de busca semântica utilizando o modelo SBERT (Sentence-BERT) da biblioteca Sentence Transformers. Esse método codifica tanto os documentos quanto as queries, calculando a similaridade de cosseno entre eles para retornar os mais relevantes.
  - A busca semântica foi testada com uma query e demonstrou ser eficaz ao identificar documentos com alto grau de similaridade, comprovando a capacidade do sistema em capturar similaridades semânticas sutis.

== Análise de Tópicos ==
  - A análise de tópicos foi conduzida com o modelo LDA (Latent Dirichlet Allocation) da Gensim, que identificou palavras-chave associadas aos principais tópicos no corpus. Isso forneceu uma visão dos temas predominantes nos textos.

== Identificação de Entidades ==
  - O spaCy foi utilizado para extração de entidades nomeadas, como pessoas, locais, organizações e outras categorias. Esse processo revelou uma variedade de entidades presentes nos textos, refletindo a diversidade de conteúdos no corpus.

== Visualizações ==
  - Heatmap de Similaridade: O heatmap exibiu a similaridade entre os documentos, com valores variando de 0.3 a 0.9. Observou-se que alguns documentos possuem temas relacionados, enquanto outros mostraram menor similaridade, destacando a variação temática do corpus.
  - Resultados de Busca Semântica: O sistema de recomendação retornou documentos relevantes com base na query fornecida, com uma similaridade significativa, validando a eficiência do modelo SBERT para busca.
  - Identificação de Entidades: A extração mostrou uma lista diversa de entidades, confirmando a riqueza informativa dos textos.

== Análise Interpretativa == 
  - O heatmap de similaridade sugere que, embora haja uma diversidade de tópicos, alguns documentos compartilham semelhanças, evidenciadas por células com valores acima de 0.7.
  - A busca semântica com SBERT mostrou que o sistema é eficaz em encontrar documentos relevantes de acordo com consultas específicas, oferecendo uma abordagem robusta para sistemas de recomendação.
  - A análise de tópicos revelou temas gerais e a extração de entidades demonstrou a presença de informações relevantes e estruturadas nos textos.

Explicação dos Blocos de Código:
  - `calcular_similaridade_embedding_spacy`: Aplica embeddings do spaCy aos textos e retorna a matriz de similaridade.
  - `busca_semantica`: Realiza a busca semântica usando SBERT e retorna os documentos mais relevantes.
  - `print_topics`: Identifica e exibe os tópicos usando LDA.
  - `identify_entities`: Extrai e lista entidades dos textos com o spaCy.
  - `plotar_heatmap`: Gera um heatmap para visualização das similaridades entre os documentos.

Conclusão:
Este código cumpre os requisitos para a aplicação de embeddings de acordo que o professor pede, incluindo cálculo de similaridade, sistemas de recomendação, análise de tópicos e identificação de entidades. As visualizações e resultados fornecem insights claros sobre a estrutura e os temas presentes no corpus, destacando a eficácia dos métodos aplicados.
"""

import gensim
import gensim.corpora
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""
Rodar para instalar as dependencias
!pip install -r requirements.txt
!python -m spacy download pt_core_news_lg
"""

# Carregar o modelo de linguagem do spaCy
nlp = spacy.load('pt_core_news_lg')
# Carregar modelo SBERT
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# Carregar nltk
nltk.download('punkt')
qdt_rows = 25


def calcular_similaridade_tf_idf(data: list[str]) -> pd.DataFrame:
    try:
        # Vetorização com TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(data)
        # Calculando a similaridade por cosseno entre os documentos
        cosine_sim = cosine_similarity(tfidf_matrix)
        return criar_dataframe_matriz(cosine_sim)
    except Exception as e:
        print(f"Erro ao calcular similaridade TF-IDF: {e}")
        return pd.DataFrame()


def calcular_similaridade_embedding_spacy(data: list[str]) -> pd.DataFrame:
    try:
        # Aplicando embedding
        embedded_data = [nlp(news) for news in data]
        # Calculando a similaridade por cosseno entre os documentos
        similaridade = [[doc1.similarity(doc2) for doc2 in embedded_data] for doc1 in embedded_data]
        return criar_dataframe_matriz(similaridade)
    except Exception as e:
        print(f"Erro ao calcular similaridade com embeddings do spaCy: {e}")
        return pd.DataFrame()


def criar_dataframe_matriz(matriz: list[list[float]]) -> pd.DataFrame:
    # Convertendo para um DataFrame para melhor visualização
    return pd.DataFrame(matriz, index=[f"Doc{i + 1}" for i in range(0, qdt_rows)],
                        columns=[f"Doc{i + 1}" for i in range(0, qdt_rows)])


def plotar_heatmap(data: pd.DataFrame, title: str):
    # Configurando o tamanho da figura
    plt.figure(figsize=(20, 16))
    # Gerando o heatmap da matriz de similaridade por cosseno
    sns.heatmap(data, annot=True, cmap='Blues', linewidths=0.5)
    # Definindo o título do heatmap
    plt.title(title)
    # Exibindo o gráfico
    plt.show()


def busca_semantica(query: str,
                    model: SentenceTransformer,
                    data: pd.DataFrame,
                    document_embeddings: np.array,
                    top_n: int = 5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, document_embeddings)

    # Encontrar os documentos mais similares à query
    top_doc_indices = np.argsort(similarities[0])[-top_n:]
    for index in reversed(top_doc_indices):
        print(f"Documento similar à query '{query}':\n{data['title'][index]}")
        print(f"Similaridade: {similarities[0][index]}\n")


def identify_entities(data: list[str]) -> set:
    # Aplicando embedding
    embedded_data = ' '.join(data)

    # Processar o texto
    doc = nlp(embedded_data)

    entities: set = set()
    # Identificar entidades
    for ent in doc.ents:
        entities.add(ent)
        print(f'Entidade: {ent.text}, Tipo: {ent.label_}')
    return entities


def print_topics(data: list[str]):
    # Tokenização
    texts = [word_tokenize(doc.lower()) for doc in data]

    # Criar um dicionário e um corpus
    dictionary = gensim.corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Criar o modelo LDA
    lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=5)

    # Mostrar os tópicos
    for idx, topic in lda_model.print_topics(-1):
        print(f'Tópico {idx}: {topic}')


if __name__ == '__main__':
    try:
        df = pd.read_csv('uol_news_data.csv', sep=';', encoding='utf-8-sig')
        data = df['content_without_pontuation'][:qdt_rows]
        tokens_lemmatized = df['tokens_lemmatized'][:qdt_rows]
        tokens_stemmed = df['tokens_stemmed'][:qdt_rows]

        # Identificacao de topicos
        print(f'\nTokens com lematização')
        print_topics(tokens_lemmatized)
        print(f'\nTokens com stemização')
        print_topics(tokens_stemmed)
        print('\n')

        # Similaridade
        similaridade_tf_idf = calcular_similaridade_tf_idf(data)
        similaridade_embedding = calcular_similaridade_embedding_spacy(data)

        plotar_heatmap(similaridade_tf_idf, 'Heatmap da Similaridade por Cosseno entre Documentos (TF-IDF)')
        plotar_heatmap(similaridade_embedding, 'Heatmap da Similaridade por Cosseno entre Documentos (SPACY)')

        # Gerar embeddings dos documentos
        doc_embeddings = sbert_model.encode(data, convert_to_numpy=True)

        # Busca semantica
        query = input("Digite os termos de busca: ")
        busca_semantica(
            query=query,
            model=sbert_model,
            data=df,
            document_embeddings=doc_embeddings,
            top_n=1
        )

        # Identifica entidades
        entities = identify_entities(data)
    except Exception as e:
        print(f"Erro ao executar o script: {e}")

