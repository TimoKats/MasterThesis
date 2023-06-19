__file__ = 'quorum.py'
__author__ = 'Timo Kats'
__description__ = 'Runs experiment for the quorum operator. Requires dataset and command line arguments.'

# libraries

import json, argparse
import pandas as pd

# local imports

from include.preprocess import *
from include.results import *
from include.data import *

# initialize parser

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--Topic", help = "Topic of the source document")
parser.add_argument("-m", "--M", help = "Precision")
parser.add_argument("-n", "--N", help = "Recall")
args = parser.parse_args()

def create_document_index(documents):
    '''
    create_document_index creates an inverted index using document identifiers.

    :param documents: a list of (pre-processed) documents
    :return: dictionary where the keys are words and the values are lists of document identifiers
    '''
    document_index = {}
    for index, article in enumerate(documents):
        for word in article:
            if word in document_index.keys():
                document_index[word].append(index)
            else:
                document_index[word] = [index]
    return document_index

class Quorum:
    def __init__(self, m, n, data):
        self.processed_documents = process_documents(list(data['text']), n)
        self.document_index = create_document_index(self.processed_documents)
        self.doc_lengths = doc_lengths(self.processed_documents)
        self.positive_set = get_positive_set(data, args.Topic)
        self.m = m
        self.n = n

    def query(self, query):
        '''
        query conducts a quorum search for a query

        :param query: (pre-processed) list of query-terms
        :return: dictionary of returned set of results
        '''
        matches = {}
        for term in self.processed_documents[query]:
            for document in self.document_index[term]:
                if document in matches.keys():
                    matches[document] += 1
                else:
                    matches[document] = 1
        return {k: v for k, v in sorted(matches.items(), key=lambda item: item[1], reverse=True)}

if __name__ == '__main__':
    data = load_reuters("../data/paragraphs/test/test_documents.csv")
    quorum_session = Quorum(float(args.M), float(args.N), data)

    for document_id in quorum_session.positive_set:
        quorum_similarities = quorum_session.query(document_id)
        quorum_similarities = sort_similarities(quorum_similarities, quorum_session.doc_lengths, quorum_session.m)
        update_results(quorum_similarities, quorum_session.positive_set)

    for cutoff, result in cutoffs.items():
        recall,precision,f1_score = get_scores(result)
        output_results(cutoff,args.Topic,precision,recall,f1_score)
