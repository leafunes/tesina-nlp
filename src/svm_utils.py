import numpy as np

from sklearn.metrics import accuracy_score


def get_best_tokens_dummy(corpus, each_q):
    pos = corpus[corpus['rate'] == 'positive']['content'].str.split(expand=True).stack().value_counts()
    neg = corpus[corpus['rate'] == 'negative']['content'].str.split(expand=True).stack().value_counts()

    best_tokens = pos.head(each_q) \
        .add(neg.head(each_q), fill_value=0)

    return best_tokens

# Transforma la sentence en un array
def transform_sentence(sentence, best_tokens, vector_size):
    s_list = sentence.split()
    selected_in_sentence = []
    for e in s_list:
        if(e in best_tokens):
            selected_in_sentence.append(e)

    out = np.zeros((vector_size))
    for i in range(min(vector_size, len(selected_in_sentence))):
        out[i] = best_tokens.index.get_loc(selected_in_sentence[i]) + 1
    return out

def get_score(clasif, data_tuples):
    
    unzipped = list(zip(*data_tuples))
    y_predicted = clasif.predict(unzipped[0])
    return accuracy_score(unzipped[1], y_predicted)

