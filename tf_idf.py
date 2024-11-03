from typing import Any

# Visualização de Dados
import matplotlib.pyplot as plt
import numpy as np
# Manipulação de Dados
import pandas as pd
import seaborn as sns
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# Vetorização de Texto e Similaridade
from sklearn.feature_extraction.text import TfidfVectorizer  # Vetorização com TF-IDF
from sklearn.metrics.pairwise import cosine_similarity  # Cálculo de similaridade cosseno

# Processamento de Linguagem Natural

"""
a. Os dados foram vetorizados com TF-IDF e foram calculadas as similaridades entre os documentos?
b. Foram analisados clusters, sendo visualizados e interpretados com PCA ou outra técnica semelhante?
"""
qdt_rows = 25


def vectorize_dataframe_tf_idf(dataframe) -> tuple[DataFrame, Any]:
    # Passo 1: Instanciar o TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Passo 2: Ajustar e transformar os dados para obter a matriz TF-IDF
    X_tfidf = tfidf_vectorizer.fit_transform(dataframe['texto_processado'])

    # Passo 3: Converter a matriz esparsa resultante para um DataFrame
    df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Definir a coluna 'documento' como índice do DataFrame
    df_tfidf.index = dataframe['documento']
    return df_tfidf, X_tfidf


def calculate_similarity_documents(dataframe, X_tfidf) -> pd.DataFrame:
    # Passo 4: Calcular a similaridade cosseno entre os documentos
    similaridade = cosine_similarity(X_tfidf)

    # Passo 5: Converter a matriz de similaridade em um DataFrame para melhor visualização
    df_similaridade = pd.DataFrame(similaridade, index=dataframe['documento'], columns=dataframe['documento'])

    return df_similaridade


def plot_similarity_heatmap(df_similaridade):
    # Ajustar o tamanho da figura
    plt.figure(figsize=(20, 16))

    # Criar o heatmap usando seaborn
    sns.heatmap(df_similaridade, annot=True, cmap='Blues', fmt=".2f", linewidths=.5)

    # Ajustes de título e rótulos
    plt.title('Similaridade entre Documentos (TF-IDF)', fontsize=16)
    plt.xlabel('Documentos', fontsize=12)
    plt.ylabel('Documentos', fontsize=12)

    # Exibir o gráfico
    plt.tight_layout()
    plt.show()


# Define os klusters
def get_klusters(df_documents, tfidf_matrix, num_clusters=3) -> pd.DataFrame:
    # Definindo o número de clusters (podemos testar com 3 inicialmente)

    # Aplicando o KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)

    # Adicionando as labels de clusters ao DataFrame original
    df_documents['Cluster'] = kmeans.labels_
    return df_documents


# Visualização de grafícos com PCA
def plot_df_kluster(df_documents, tfidf_matrix, num_clusters=3):
    # Reduzindo a dimensionalidade para 3 componentes principais
    pca = PCA(n_components=3)
    reduced_features = pca.fit_transform(tfidf_matrix.toarray())

    # Plotando os clusters em um gráfico 3D
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')

    # Usando um mapa de cores com base no número de clusters
    cmap = plt.get_cmap('tab10')  # Escolha um mapa de cores (ex.: 'tab10', 'viridis', 'plasma', etc.)
    colors = [cmap(i) for i in np.linspace(0, 1, num_clusters)]

    # Plotando os pontos e colorindo de acordo com os clusters
    for i in range(num_clusters):
        points = reduced_features[df_documents['Cluster'] == i]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=[colors[i]], label=f'Cluster {i + 1}')

    ax.set_title('Document Clusters (reduzido para 3D via PCA)')
    ax.set_xlabel('PCA Component 1')
    ax.set_zlabel('PCA Component 2')
    ax.set_ylabel('PCA Component 3')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    # Comparação de apenas 25 documentos, se todos os documentos fossem utilizados o gráfico ficaria muito poluído e ilegível
    df = pd.read_csv(r'textos_processados.csv')[:qdt_rows]

    # Utilizando texto sem pontuação
    df['texto_processado'] = pd.read_csv(
        'uol_news_data.csv',
        sep=';',
        encoding='utf-8-sig')['content_without_pontuation'][:qdt_rows]

    df_tfidf, X_tfidf = vectorize_dataframe_tf_idf(df)

    df_similarity = calculate_similarity_documents(df, X_tfidf)

    plot_similarity_heatmap(df_similarity)

    df_kluster = get_klusters(df_tfidf, X_tfidf)

    plot_df_kluster(df_kluster, X_tfidf)
