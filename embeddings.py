"""
Bibliotecas para instalar:
pip install numpy==1.24.3 sentence-transformers spacy
python -m spacy download pt_core_news_lg

a. Foram aplicados word embeddings e técnicas relacionadas?
similaridade (SPACY) -> SIM
sistemas de recomendação ou busca -> Busca implementada
análise de tópicos
identificação de entidades
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
