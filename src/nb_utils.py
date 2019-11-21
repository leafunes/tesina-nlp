from sklearn.utils import shuffle

def get_best_tokens_dummy(corpus, each_q):
    pos = corpus[corpus['rate'] == 'positive']['content'].str.split(expand=True).stack().value_counts()
    neg = corpus[corpus['rate'] == 'negative']['content'].str.split(expand=True).stack().value_counts()

    best_tokens = pos.head(each_q) \
        .add(neg.head(each_q), fill_value=0)

    return best_tokens