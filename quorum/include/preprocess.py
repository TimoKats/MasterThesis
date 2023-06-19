__file__ = 'include/preprocess.py'
__author__ = 'Timo Kats'
__description__ = 'Various functions related to preprocessing documents.'

import re, collections

def stopwords():
    '''
    stopwords returns a list of stopwords (filename set)

    :return: list with stopwords
    '''
    file = open('../data/stopwords.txt', "r")
    stopwords = file.read()
    return stopwords.split("\n")

def n_most_common(text, n):
    '''
    n_most_common sorts words in string by frequency and returns top n

    :param text: string of natural language 
    :param n: the amount of most common words
    :return: list of sorted words
    '''
    text = collections.Counter(text).most_common()
    return [i[0] for i in text][:int(n*len(text))]

def doc_lengths(processed_documents):
    '''
    doc_lengths returns the length (ito terms) per document

    :param processed_documents: list of documents that are already pre-processed
    :return: dictionary of document lengths (length per document identifier)
    '''
    doc_lengths = {}
    for index, text in enumerate(processed_documents):
        doc_lengths[index] = len(text)
    return doc_lengths

def process_documents(content, n):
    '''
    process documents filters stopwords and sorts by n most common

    :param content: list of document contents (text)
    :param n: refers to the (n) most common words that's returned
    :return: list of documents that are preprocesed
    '''
    words = []
    for index, item in enumerate(content):
        words.append(re.split(r"[\b\W\b]+", item))
        words[index] = [x for x in words[index] if (x not in stopwords())]
        words[index] = n_most_common(words[index], n)
    return words
