"""
TF-IDF

Alunos: Rodrigo K. Franco, Maria Eduarda Krutzsch, Luan L. Guarnieri, Nicole B., Gustavo Baroni, Ana Carolina

Critérios de Avaliação:
Os dados foram vetorizados com TF-IDF e foram calculadas as similaridades entre os documentos? (Peso: 0,1)
Foram analisados clusters, sendo visualizados e interpretados com PCA ou outra técnica semelhante? (Peso: 0,1)

Descrição das Implementações:
== Vetorização com TF-IDF ==
  - Foi utilizado o `TfidfVectorizer` da biblioteca `scikit-learn` para transformar o texto processado em uma matriz TF-IDF. Essa matriz representa a importância de cada termo em relação a um documento específico e ao corpus como um todo.
  - A similaridade entre os documentos foi calculada utilizando a 'similaridade cosseno', uma técnica que mede a similaridade entre dois vetores com base no ângulo entre eles.

== Cálculo de Similaridade (item 1) ==
  - A matriz de similaridade cosseno foi gerada e visualizada em um 'heatmap', onde cada célula representa a similaridade entre dois documentos. A diagonal principal tem sempre o valor 1, indicando que cada documento é 100% similar a si mesmo.

== Análise de Clusters (item 2) ==
  - A análise de clusters foi realizada utilizando o algoritmo 'K-Means' para agrupar os documentos em categorias diferentes com base em suas representações TF-IDF.
  - A visualização dos clusters foi feita com **PCA (Análise de Componentes Principais)**, reduzindo a dimensionalidade para 3 componentes principais e plotando os resultados em um gráfico 3D para facilitar a interpretação visual dos agrupamentos.

== Visualizações ==
  - Heatmap de Similaridade: O heatmap apresenta a similaridade entre os documentos, com valores variando entre 0 e 1. Observa-se que a maioria dos documentos tem similaridade entre 0.1 e 0.3, indicando uma diversidade temática no corpus. Algumas células com valores mais altos (próximas da diagonal) mostram documentos que compartilham temas semelhantes.
  - Gráfico 3D de Clusters com PCA: Mostra a distribuição dos documentos em 3 clusters. Pontos próximos indicam documentos com conteúdo semelhante. Alguns clusters são bem definidos, enquanto outros apresentam pontos mais dispersos, indicando possíveis interseções de temas ou documentos com menos coesão.

== Análise Interpretativa == 
  - O heatmap de similaridade confirma a diversidade de temas no corpus, com poucos documentos apresentando alta similaridade entre si. Isso sugere uma base de dados com ampla variedade de tópicos.
  - O gráfico 3D dos clusters revela agrupamentos com algumas sobreposições, mostrando que alguns documentos compartilham características temáticas, enquanto outros formam grupos mais homogêneos. Isso pode indicar tanto a presença de documentos com temas mistos quanto clusters bem definidos em alguns casos.

Explicação dos Blocos de Código:
  - `vectorize_dataframe_tf_idf`: Vetoriza os textos utilizando TF-IDF e retorna um DataFrame com a matriz TF-IDF e a matriz esparsa original.
  - `calculate_similarity_documents`: Calcula a similaridade cosseno entre os documentos e retorna um DataFrame com os valores.
  - `plot_similarity_heatmap`: Plota um heatmap para visualização das similaridades entre os documentos.
  - `get_klusters`: Aplica o algoritmo K-Means aos documentos vetorizados e adiciona rótulos de clusters ao DataFrame.
  - `plot_df_kluster`: Plota um gráfico 3D dos clusters utilizando PCA para reduzir a dimensionalidade.

Conclusão:
Este código atende aos critérios do TF-IDF e da análise de clusters passados pelo professor, ele inclui a vetorização dos textos com TF-IDF, cálculo da similaridade cosseno, e visualizações que ajudam a interpretar as similaridades e os agrupamentos de documentos. As análises revelam que o corpus possui uma diversidade de tópicos e que os documentos foram agrupados de forma coerente em alguns clusters, com algumas sobreposições indicando temas compartilhados.

O primeiro gráfico de Heatmap mostra a similaridade entre os documentos. Cada célula representa a similaridade entre dois documentos, sendo a diagonal principal,
a similaridade a si mesmo, totalizando em 1 corretamente, ou seja, cada documento é 100% similar a si mesmo. O heatmap apresenta poucos resultados com similaridade,
sendo a maioria entre 0.1 a 0.3, o que sugere que o "corpus" é composto de temas bem distintos entre si, ou seja, existe uma diversidade de tópicos. Entretanto, há algumas células próximas da diagonal que
apresentam valores mais altos de similaridade (destacadas em tons mais escuros, com valores acima de 0.3), indicando que os documentos compartilham um conteúdo ou tema semelhante.

O segundo gráfico da visualização 3d com PCA, representa os documentos agrupados em 3 clusters. A presença de pontos próximos indicam documentos que compartilham
similaridades. Alguns clusters são bem definidos, enquanto outros têm pontos dispersos e sobreposições, indicando temas compartilhados ou falta de coesão entre os documentos. 

"""

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
    df_25 = pd.read_csv(r'textos_processados.csv')[:25]

    # Utilizando texto sem pontuação
    df_25['texto_processado'] = pd.read_csv(
        'uol_news_data.csv',
        sep=';',
        encoding='utf-8-sig')['content_without_pontuation'][:25]

    df_25_tfidf, X_25_tfidf = vectorize_dataframe_tf_idf(df_25)

    df_similarity = calculate_similarity_documents(df_25, X_25_tfidf)

    plot_similarity_heatmap(df_similarity)

    df = pd.read_csv(r'textos_processados.csv')[:200]

    # Utilizando texto sem pontuação
    df['texto_processado'] = pd.read_csv(
        'uol_news_data.csv',
        sep=';',
        encoding='utf-8-sig')['content_without_pontuation'][:200]

    df_tfidf, X_tfidf = vectorize_dataframe_tf_idf(df)

    df_kluster = get_klusters(df_tfidf, X_tfidf)

    plot_df_kluster(df_kluster, X_tfidf)
