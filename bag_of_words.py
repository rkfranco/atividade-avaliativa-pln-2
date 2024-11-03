# Visualização de Dados
import matplotlib.pyplot as plt
import numpy as np
# Manipulação de Dados
import pandas as pd
import seaborn as sns
from matplotlib.widgets import Slider
# Vetorização de Texto e Similaridade
from sklearn.feature_extraction.text import CountVectorizer  # Vetorização com Bag of Words (BoW)

"""
Os dados foram vetorizados com BoW e foram calculados termos
mais frequentes do corpus, incluindo análises com visualizações?
"""


def vectorize_dataframe_bow(dataframe) -> pd.DataFrame:
    # Instanciar o CountVectorizer
    vectorizer = CountVectorizer()

    # texto_sem_nan = [doc for doc in df['texto_processado'] if doc is not np.nan]

    # Aplicar a vetorização ao texto processado
    X_bag_of_words = vectorizer.fit_transform(dataframe['texto_processado'])

    # Converter a matriz esparsa resultante para um DataFrame
    df_bag_of_words = pd.DataFrame(X_bag_of_words.toarray(), columns=vectorizer.get_feature_names_out())

    # Definir a coluna 'documento' como índice do DataFrame
    df_bag_of_words.index = dataframe['documento']

    # Exibir as primeiras linhas do DataFrame vetorizado com 'documento' como índice
    return df_bag_of_words


def show_total_tokens_per_document(df_bag_of_words):
    # Configurações de estilo e paleta de cores
    sns.set_style("whitegrid")  # Estilo de grade branca
    sns.set_palette("viridis")  # Paleta de cores 'viridis'

    # Passo 1: Calcular o total de tokens em cada documento
    total_tokens = df_bag_of_words.sum(axis=1)

    # Criar um DataFrame para facilitar a plotagem
    df_total_tokens = total_tokens.reset_index()
    df_total_tokens.columns = ['Documento', 'Total de Tokens']

    # Ordenar os documentos pela frequência de tokens em ordem decrescente
    df_total_tokens = df_total_tokens.sort_values(by='Total de Tokens', ascending=False)

    # Criar a figura e os eixos
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)

    # Visualização do total de tokens por documento
    barplot = sns.barplot(x='Documento', y='Total de Tokens', data=df_total_tokens, ax=ax)

    # Adicionar os valores no topo das barras
    for index, row in enumerate(df_total_tokens['Total de Tokens']):
        barplot.text(index, row, f'{int(row)}', color='black', ha="center", va="bottom")

    # Ajustes de título e rótulos
    plt.title('Total de Tokens por Documento')
    plt.title('Total de Tokens por Documento')
    plt.xlabel('Documento')
    plt.ylabel('Total de Tokens')
    plt.xticks(rotation=90)  # Rotaciona os rótulos do eixo X para melhor legibilidade

    # Slider para rolagem do eixo X
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Documento', 0, len(df_total_tokens) - 10, valinit=0, valstep=1)

    # Função para atualizar os limites do eixo X
    def update(val):
        start = int(slider.val)
        end = start + 10  # Mostra 10 documentos de cada vez
        ax.set_xlim(start, end)
        ax.set_xticks(np.arange(start, end))  # Atualiza os ticks do eixo X
        ax.set_xticklabels(df_total_tokens['Documento'].iloc[start:end], rotation=90)  # Atualiza os rótulos
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.tight_layout()

    # Exibir o gráfico
    plt.show()


def get_most_frequent_tokens_per_doc(df_bow, top_n=10) -> pd.DataFrame:
    top_tokens_list = [
        {'documento': documento, 'token': token, 'frequencia': count}
        for documento, row in df_bow.iterrows()
        for token, count in row.nlargest(top_n).items()  # Obtém os N tokens mais frequentes
        if count > 0  # Garante que os tokens tenham contagem maior que zero
    ]
    # Converter a lista em DataFrame
    df_top = pd.DataFrame(top_tokens_list)
    return df_top


def get_most_frequent_tokens(df_bow, top_n=10) -> pd.DataFrame:
    row = df_bow.sum(axis=0)
    top_tokens_list = [
        {'token': token, 'frequencia': count}
        for token, count in row.nlargest(top_n).items()  # Obtém os N tokens mais frequentes
        if count > 0  # Garante que os tokens tenham contagem maior que zero
    ]
    # Converter a lista em DataFrame
    df_top = pd.DataFrame(top_tokens_list)
    return df_top


def plot_most_frequent_tokens(df_most_frequent_tokens):
    plt.figure(figsize=(16, 12))
    plt.bar(
        df_most_frequent_tokens['token'],
        df_most_frequent_tokens['frequencia'],
        color='skyblue'
    )

    plt.title('Tokens mais frequentes do Corpus')
    plt.xlabel('Tokens')
    plt.ylabel('Frequência')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv(r'textos_processados.csv')
    df_bow = vectorize_dataframe_bow(df)
    # Clique no scroll horizontal para melhor visualização dos dados
    show_total_tokens_per_document(df_bow)
    # Calcula os tokens mais frequentes por documento (Não está sendo exibido devido ao tamanho do dataset)
    df_most_frequent_tokens_per_doc = get_most_frequent_tokens_per_doc(df_bow, 50)
    # Calcula os tokens mais frequentes do Corpus inteiro
    df_most_frequent_tokens = get_most_frequent_tokens(df_bow, 25)
    plot_most_frequent_tokens(df_most_frequent_tokens)
