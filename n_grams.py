import pandas as pd
from nltk import ngrams


def get_ngrams(n: int) -> list:
    return [list(ngrams(doc.split(), n)) for doc in df['texto_processado']]


if __name__ == '__main__':
    qdt_rows = 25
    df = pd.read_csv(r'textos_processados.csv')[:qdt_rows]
    documents = df['texto_processado']

    dados = {
        'documento': df['documento'],
        'n_grams_1': get_ngrams(1),
        'n_grams_2': get_ngrams(2),
        'n_grams_3': get_ngrams(3),
        'n_grams_4': get_ngrams(4),
        'n_grams_5': get_ngrams(5),
    }

    pd.DataFrame(dados).to_csv('n_grams.csv')
