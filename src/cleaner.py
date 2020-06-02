from logger import log, debug

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import nltk
from fillerwords import fillerWords
from nltk.tag.stanford import StanfordPOSTagger


from multiprocessing import cpu_count, Pool
import numpy as np
import pandas as pd
import re
 
cores = cpu_count() #Number of CPU cores on your system
partitions = 8 #Define as many partitions as you want

# --------------- Mapeo de strings --------------
stemmer = SnowballStemmer("spanish")
re_points = re.compile("\\.")
re_symbols = re.compile("\\,|\\-|\\!|\\¡|\\(|\\)")
re_points_and_spaces = re.compile(" |\\.")

path_to_model = "bin/models/spanish-ud.tagger"
path_to_jar = "bin/stanford-postagger.jar"

tagger = StanfordPOSTagger(model_filename=path_to_model, path_to_jar=path_to_jar)

def sentiment_to_vector(sentiment):
    if sentiment is 'positive':
        return np.array([1, 0])
    else:
        return np.array([0, 1])


def clean_series(series):
    return series.map(clean_sentence)

def clean_sentence(sentence):
    newSentenceAsList = []
    upper = sentence.upper()
    clean = re_symbols.sub("", upper)
    clean = re_points.sub(" ", clean)
    tokens = nltk.word_tokenize(clean, 'spanish')
    for token in tokens:
        stemed = stemmer.stem(token).upper()
        if token not in fillerWords and stemed not in fillerWords:
            newSentenceAsList.append(stemed)
    return " ".join(newSentenceAsList)

def clean_series_standford(series):
    return series.map(clean_sentence_standford)

def clean_sentence_standford(sentence):
    upper = sentence.upper()
    clean = re_symbols.sub("", upper)
    clean = re_points.sub(" ", clean)

    tagged = tagger.tag(re_points_and_spaces.split(clean))

    newSentenceAsList = []
    tokens = []
    for (token, token_type) in tagged:
        if token_type in ["ADJ", "NOUN", "VERB"]:
            tokens.append(token)
    for token in tokens:
        stemed = stemmer.stem(token).upper()
        if token not in fillerWords and stemed not in fillerWords:
            newSentenceAsList.append(stemed)
    return " ".join(newSentenceAsList)

def parallelize(data, func):
    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def clean_corpus(df_corpus, cleaner):
    debug("[Limpiando el corpus...]")
     
    debug("[Usando " + str(partitions) + " threads ...]")

    cleaned = parallelize(df_corpus["content"], cleaner)
    newCorpus = pd.DataFrame({'content': cleaned,
                            'sentiment': df_corpus['rate'].map(sentiment_to_vector),
                            'raw': df_corpus["content"],
                            'rate': df_corpus["rate"]})
    return newCorpus

def clean_corpus_neutral(df_corpus, cleaner):
    debug("[Limpiando el corpus...]")
     
    debug("[Usando " + str(partitions) + " threads ...]")

    cleaned = parallelize(df_corpus["raw"], cleaner)
    newCorpus = pd.DataFrame({'content': cleaned,
                            'raw': df_corpus["raw"],
                            'human_rate': df_corpus["human_rate"]})
    return newCorpus

def clean_corpus_human(df_corpus, cleaner):
    debug("[Limpiando el corpus clasificado por humanos...]")
     
    debug("[Usando " + str(partitions) + " threads ...]")

    cleaned = parallelize(df_corpus["raw"], cleaner)
    newCorpus = pd.DataFrame({'content': cleaned,
                            'sentiment': df_corpus['human_rate'].map(sentiment_to_vector),
                            'raw': df_corpus["raw"],
                            'rate': df_corpus["rate"],
                            'human_rate': df_corpus["human_rate"]})
    return newCorpus

def clean_corpus_basic(df_corpus):
    debug("[Usando cleaner basico]")
    return clean_corpus(df_corpus, clean_series)

def clean_corpus_standford(df_corpus):
    debug("[Usando pos-tagger]")
    return clean_corpus(df_corpus, clean_series_standford)

def clean_corpus_basic_neutral(df_corpus):
    debug("[Usando cleaner basico para samples neutrales]")
    return clean_corpus_neutral(df_corpus, clean_series)
    
def clean_corpus_basic_human(df_corpus):
    debug("[Usando cleaner basico para samples clasificados por humanos]")
    return clean_corpus_human(df_corpus, clean_series)

