__file__ = 'create_embeddings.py'
__author__ = 'Timo Kats'
__description__ = 'Creates TFIDF embeddings for the SVM experiments.'

# libraries

import re, argparse, json, pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# initialize parser

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--Stopwords", help = "Do you want stopwords?")
args = parser.parse_args()

# I/O functions

def ouput_embeddings(embeddings, args):
    filename = 'embeddings/svm.npy'
    with open(filename, 'wb') as file:
        np.save(file, embeddings)
    file.close()

def load_stopwords():
    if args.Stopwords:
        file = open('../data/stopwords.txt', "r")
        stopwords = file.read()
        return stopwords.split("\n")
    else:
        return None

def load_reuters():
    test = pd.read_csv('../data/svm/svm_test.csv', sep=';')
    train = pd.read_csv('../data/svm/svm_train.csv', sep=';')
    return list(train['text']) + list(test['text'])

if __name__ == '__main__':
    args = parser.parse_args()
    content = load_reuters()

    TFIDF_vectorizer = TfidfVectorizer(stop_words=load_stopwords(), token_pattern=r"(?u)\b\w+\b")
    embeddings = TFIDF_vectorizer.fit_transform(content).toarray()
    ouput_embeddings(embeddings, args)

