__file__ = 'keywords_feedback.py'
__author__ = 'Timo Kats'
__description__ = 'Conducts relevance feedback experiments for keywords.'

# libraries

import pysolr, argparse, numpy as np, pandas as pd
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer

# solr

solr = pysolr.Solr('http://localhost:8983/solr/ms-marco')

# initialize parser

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--Topic", help = "Topic with the source documents")
args = parser.parse_args()

# I/O functions

def export(result):
    filename = 'results/ambiguous.json'
    with open(filename,'a+', encoding='utf-8') as f:
        f.write(str(round(np.mean(result),3)) + ';' + str(round(np.std(result),3)) + '\n')

# other

def get_embeddings():
    query_data = {"docId":[], "embedding":[]}
    queries = solr.search('topic:(' + args.Topic + ')', **{'rows':5000})
    for query in queries:
        query_data['docId'].append(query['docId'])
        query_data['embedding'].append(query['bertbase'])
    query_df = pd.DataFrame.from_dict(query_data)
    return list(query_df.groupby('docId').head(n=1).reset_index(drop=True)['embedding'])

def get_maxsize():
    documents = set()
    results = solr.search('topic:(' + args.Topic + ')', **{'rows':5000, 'fl':'docId'})
    for result in results:
        documents.add(result['docId'])
    return len(documents)

# TFIDF functions

def get_stopwords():
    file = open('../data/stopwords.txt', "r")
    stopwords = file.read()
    return stopwords.split("\n")

def process_feedback(user_feedback):
    try:
        tf = TfidfVectorizer(use_idf=True, norm=None, stop_words=get_stopwords())
        tf.fit_transform(user_feedback)
        return sorted(tf.vocabulary_.items(), key=lambda item: item[0])[:10]
    except:
        return []

class KeywordFeedback:
    def __init__(self, embedding):
        self.collected_documents = set()
        self.declined_documents = set()
        self.stopwords = get_stopwords()

        self.embedding = embedding
        self.user_feedback = []
        self.keywords = []

        self.size = get_maxsize()
        self.iterations = 1

    def format_keyword_query(self):
        query = ""
        for word in self.keywords:
            query += 'text:' + str(word[0]) + ' OR '
        return '(' + query[:-4] + ')'

    def filter_docs(self):
        filter = ''
        results_keyword = solr.search(self.format_keyword_query(), **{'rows':7500, 'fl':'docId'})
        for result_keyword in results_keyword:
            if result_keyword['docId'] not in filter:
                filter += ' ' + result_keyword['docId'] + ' '
        return filter

    def prefilter(self):
        filter = ""
        for docId in chain(self.collected_documents, self.declined_documents):
            filter += ' ' + str(docId) + ' '
        try:
            return 'NOT docId: (' + filter + ') AND docId: (' + self.filter_docs() + ')'
        except:
            return 'NOT docId: (' + filter + ')'

    def do_query(self):
        if len(self.collected_documents) > 0:
            return solr.search('{!knn f=bertbase topK=10}' + str(self.embedding),**{'rows':10, 'fq':self.prefilter()})
        else:
            return solr.search('{!knn f=bertbase topK=10}' + str(self.embedding), **{'rows':10})

    def iterate(self):
        results = self.do_query()
        user_feedback = []
        for result in results:
            if args.Topic in result['topic']:
                self.collected_documents.add(result['docId'])
                user_feedback.append(result['text'][0])
            else:
                self.declined_documents.add(result['docId'])
        self.keywords += process_feedback(user_feedback)

    def recall(self):
        return len(self.collected_documents) / self.size

# main

if __name__ == '__main__':
    result = []
    for embedding in get_embeddings():
        keyword_session = KeywordFeedback(embedding)
        while keyword_session.recall() < 0.8:
            keyword_session.iterate()
            keyword_session.iterations += 1
        result.append(keyword_session.iterations)
    export(result)
