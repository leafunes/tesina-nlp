from logger import log, debug

from sklearn.utils import shuffle

import numpy as np

# --------------- Feature Extractor --------------
# ---------- Transformacion de vector ------------

# Selecciona los best_tokens, en base a cantidad de apariciones en cada label
def get_best_tokens_dummy(corpus, each_q):
    pos = corpus[corpus['rate'] == 'positive']['content'].str.split(expand=True).stack().value_counts()
    neg = corpus[corpus['rate'] == 'negative']['content'].str.split(expand=True).stack().value_counts()

    best_tokens = pos.head(each_q) \
        .add(neg.head(each_q), fill_value=0)

    best_tokens = shuffle(best_tokens)

    return best_tokens

# Transforma la sentence en un array bidimencional de 0s y 1s
def transform_sentence(sentence, best_tokens, vector_size):
    s_list = sentence.split()
    selected_in_sentence = []
    for e in s_list:
        if(e in best_tokens.index):
            selected_in_sentence.append(best_tokens.index.get_loc(e))

    out = np.zeros((vector_size, best_tokens.size))
    for i in range(min(vector_size, len(selected_in_sentence))):
        out[i][selected_in_sentence[i]] = 1
    return out

# Devuelve una funcion que tokeniza una serie de sentences, en base a los best_tokens
def get_tokenizer(best_tokens, vector_size):

    def curry(sentence_series):
    
        return sentence_series.map(lambda x: transform_sentence(x, best_tokens, vector_size))
  
    return curry
