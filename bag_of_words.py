"""
Bag of Words (BoW)

Alunos: Rodrigo K. Franco, Maria Eduarda Krutzsch, Luan L. Guarnieri, Nicole B., Gustavo Baroni, Ana Carolina

Critérios de Avaliação:
Os dados foram vetorizados com BoW e foram calculados termos mais frequentes do corpus, incluindo análises com visualizações? (Peso: 0,1)

Descrição das Implementações:
== Vetorização com BoW ==
  - Foi utilizado `CountVectorizer` da biblioteca `scikit-learn` para transformar o texto em um vetor de contagem de palavras. O resultado foi um DataFrame que representa a frequência de termos em cada documento.

== Cálculo dos Termos Mais Frequentes ==
  - As funções `get_most_frequent_tokens_per_doc` e `get_most_frequent_tokens` para identificar, respectivamente, os N tokens mais frequentes por documento e no corpus completo. Isso permite uma análise detalhada e uma visão global das palavras mais comuns.

== Visualizações ==
   - A função `show_total_tokens_per_document` gera um gráfico de barras que mostra o total de tokens em cada documento. Essa visualização ajuda a identificar a variação no tamanho dos textos e os documentos mais extensos.
   - A função `plot_most_frequent_tokens` cria um gráfico de barras destacando os tokens mais frequentes em todo o corpus, evidenciando os temas predominantes.

== Análise Interpretativa == 
   - As visualizações mostram que alguns documentos têm um número significativamente maior de tokens, enquanto a maioria possui menos palavras. O gráfico de tokens mais frequentes destaca palavras como "marçal", "governo" e "presidente", indicando um possível foco em conteúdos políticos, possivelmente relacionados às eleições no Brasil durante o mês de outubro.
   - A diferença de frequência entre os tokens mais comuns é relativamente pequena, sugerindo uma distribuição uniforme dos termos mais frequentes no corpus.

Explicação dos Blocos de Código
  - vectorize_dataframe_bow: Vetoriza os textos utilizando BoW e retorna um DataFrame com a contagem de palavras por documento.
  - show_total_tokens_per_document: Plota um gráfico de barras mostrando o total de tokens em cada documento, permitindo a análise da distribuição de tamanhos.
  - get_most_frequent_tokens_per_doc: Calcula os tokens mais frequentes em documentos individuais para uma análise detalhada de cada texto.
  - get_most_frequent_tokens: Identifica os tokens mais frequentes em todo o corpus, proporcionando uma visão global das palavras mais comuns.
  - plot_most_frequent_tokens: Gera um gráfico que visualiza os tokens mais frequentes em todo o corpus.

Conclusão:
Este código atende os critérios de avaliação do Bag of Words descrito pelo professor, inclui vetorização do corpus, cálculo de termos mais frequentes, 
e visualizações que auxiliam na análise e interpretação dos resultados. As visualizações fornecem insights sobre a composição dos documentos e os temas predominantes no corpus.

  O primeiro gráfico ao executar o script "total de tokens por documento", indica que alguns documentos tem textos muito longos, e a maioria com 
uma quantidade menor de tokens. Apesar da grande quantidade de dados, é possível por meio da barra de rolagem gerada na figura, verificar informações específicas de cada documento, sendo no eixo x o documento 
e no eixo y, o total de tokens. 

  O segundo gráfico que exibe os tokens mais frequentes no texto como um todo. O gráfico traz com destaque termos como "marçal", "governo", e "presidente", 
o que indica um possível foco em conteúdos políticos. É possível afirmar que isto está diretamente ligado ao contexto do mês de outubro por exemplo, que é quando o trabalho foi produzido, sendo relacionado
às eleições para prefeito no Brasil. A diferença de frequência entre os tokens no topo não é muito grande, o que sugere uma distribuição uniforme.

"""

# Visualização de Dados
import matplotlib.pyplot as plt
import numpy as np
# Manipulação de Dados
import pandas as pd
import seaborn as sns
from matplotlib.widgets import Slider
# Vetorização de Texto e Similaridade
from sklearn.feature_extraction.text import CountVectorizer  # Vetorização com Bag of Words (BoW)

def vectorize_dataframe_bow(dataframe) -> pd.DataFrame:
    # instanciar o CountVectorizer
    vectorizer = CountVectorizer()

    # texto_sem_nan = [doc for doc in df['texto_processado'] if doc is not np.nan]

    # aplicando a vetorização ao texto processado
    X_bag_of_words = vectorizer.fit_transform(dataframe['texto_processado'])

    # converte a matriz esparsa resultante para um DataFrame
    df_bag_of_words = pd.DataFrame(X_bag_of_words.toarray(), columns=vectorizer.get_feature_names_out())

    # define a coluna 'documento' como índice do DataFrame
    df_bag_of_words.index = dataframe['documento']

    # exibindo as primeiras linhas do DataFrame vetorizado com 'documento' como índice
    return df_bag_of_words

""" Estava usando, qualquer coisa só voltar esse e tirar o de baixo
def show_total_tokens_per_document(df_bag_of_words):
    # Configurações de estilo e paleta de cores
    sns.set_style("whitegrid")  # Estilo de grade branca
    sns.set_palette("viridis")  # Paleta de cores 'viridis'

    # Passo 1: Calcular o total de tokens em cada documento
    total_tokens = df_bag_of_words.sum(axis=1)

    # criar um DataFrame para facilitar a plotagem
    df_total_tokens = total_tokens.reset_index()
    df_total_tokens.columns = ['Documento', 'Total de Tokens']

    # ordenar os documentos pela frequência de tokens em ordem decrescente
    df_total_tokens = df_total_tokens.sort_values(by='Total de Tokens', ascending=False)

    # criando a figura e os eixos
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

   # plt.tight_layout() -- erro de ajuste automatico, comentado para rodar no colab
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

    # Exibir o gráfico
    plt.show()
"""

def show_total_tokens_per_document(df_bag_of_words, num_docs=30):
    # Mostra os num_docs documentos com mais tokens para melhor visualização
    sns.set_style("whitegrid")  # Define o estilo do gráfico como uma grade branca
    sns.set_palette("viridis")  # Define a paleta de cores como 'viridis'

    # Calcula o total de tokens em cada documento somando as colunas (contagem de palavras)
    total_tokens = df_bag_of_words.sum(axis=1)

    # Cria um DataFrame para facilitar a plotagem
    df_total_tokens = total_tokens.reset_index()
    df_total_tokens.columns = ['Documento', 'Total de Tokens']  # Renomeia as colunas para melhor entendimento

    # Ordena os documentos pelo total de tokens em ordem decrescente e seleciona os num_docs mais longos
    df_total_tokens = df_total_tokens.sort_values(by='Total de Tokens', ascending=False).head(num_docs)

    # Cria uma figura de tamanho definido
    plt.figure(figsize=(14, 7))

    # Plota um gráfico de barras do total de tokens por documento
    barplot = sns.barplot(x='Documento', y='Total de Tokens', data=df_total_tokens)

    # Adiciona rótulos com o valor no topo de cada barra
    for index, row in enumerate(df_total_tokens['Total de Tokens']):
        barplot.text(index, row, f'{int(row)}', color='black', ha="center", va="bottom")

    # Define o título e os rótulos dos eixos
    plt.title('Top Documentos por Total de Tokens')
    plt.xlabel('Documento')
    plt.ylabel('Total de Tokens')

    # Rotaciona os rótulos do eixo X para melhorar a legibilidade
    plt.xticks(rotation=90)

    # Ajusta o layout para evitar sobreposição dos elementos
    plt.tight_layout()

    # Exibe o gráfico
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

