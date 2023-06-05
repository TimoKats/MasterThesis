__file__ = 'include/preprocess.py'
__author__ = 'Timo Kats'
__description__ = 'Various functions related to preprocessing documents.'

import re, collections

def stopwords():
    file = open('../data/stopwords.txt', "r")
    stopwords = file.read()
    return stopwords.split("\n")

def n_most_common(text, n):
    text = collections.Counter(text).most_common()
    return [i[0] for i in text][:int(n*len(text))]

def doc_lengths(processed_documents):
    doc_lengths = {}
    for index, text in enumerate(processed_documents):
        doc_lengths[index] = len(text)
    return doc_lengths

def process_documents(content, n):
    words = []
    for index, item in enumerate(content):
        words.append(re.split(r"[\b\W\b]+", item))
        words[index] = [x for x in words[index] if (x not in stopwords())]
        words[index] = n_most_common(words[index], n)
    return words
