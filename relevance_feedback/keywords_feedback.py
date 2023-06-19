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

def export(result) -> None:
    '''
    export outputs the results to csv file

    :param result: list of iterations needed to achieve 80% recall
    :return: None
    '''
    filename = 'results/ambiguous.json'
    with open(filename,'a+', encoding='utf-8') as f:
        f.write(str(round(np.mean(result),3)) + ';' + str(round(np.std(result),3)) + '\n')

# other

def get_embeddings() -> list:
    '''
    get_embeddings returns the SBERT embeddings of the queries

    :return: list with embeddings (either first or random paragraphs)
    '''
    query_data = {"docId":[], "embedding":[]}
    queries = solr.search('topic:(' + args.Topic + ')', **{'rows':7000})
    for query in queries:
        query_data['docId'].append(query['docId'])
        query_data['embedding'].append(query['bertbase'])
    query_df = pd.DataFrame.from_dict(query_data)
    return list(query_df.groupby('docId').head(n=1).reset_index(drop=True)['embedding'])

def get_maxsize() -> int:
    '''
    get_maxsize returns the exact number of documents of a topic (almost always 300 in our case, but still)

    :return: integer with amount of documents
    '''
    documents = set()
    results = solr.search('topic:(' + args.Topic + ')', **{'rows':5000, 'fl':'docId'})
    for result in results:
        documents.add(result['docId'])
    return len(documents)

# TFIDF functions

def get_stopwords() -> list:
    '''
    get_stopwords returns a list of stopwords from a text file

    :return: list of stopwords
    '''
    file = open('../data/stopwords.txt', "r")
    stopwords = file.read()
    return stopwords.split("\n")

def process_feedback(user_feedback) -> list:
    '''
    process_feedback returns 10 keywords to be added to the keywords expansion

    :param user_feedback: text of selected documents
    :return: list of keywords based on IDF values
    '''
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

    def format_keyword_query(self) -> str:
        '''
        format_keyword_query formats the keyword expansion terms as a Solr filter

        :return: string that contains filter based on keyword expansion terms
        '''
        query = ""
        for word in self.keywords:
            query += 'text:' + str(word[0]) + ' OR '
        return '(' + query[:-4] + ')'

    def filter_docs(self) -> str:
        '''
        filter_docs creates a filter of documents that have already been approved/declined

        :return: string that contains filter based on approved/declined documents
        '''
        filter = ''
        results_keyword = solr.search(self.format_keyword_query(), **{'rows':7500, 'fl':'docId'})
        for result_keyword in results_keyword:
            if result_keyword['docId'] not in filter:
                filter += ' ' + result_keyword['docId'] + ' '
        return filter

    def prefilter(self) -> str:
        '''
        prefilter applies the filter created in filter_docs

        :return: string that contains filter based on approved/declined documents formatted for Solr
        '''
        filter = ""
        for docId in chain(self.collected_documents, self.declined_documents):
            filter += ' ' + str(docId) + ' '
        try:
            return 'NOT docId: (' + filter + ') AND docId: (' + self.filter_docs() + ')'
        except:
            return 'NOT docId: (' + filter + ')'

    def do_query(self) -> pysolr.Results:
        '''
        do_query executes the DVS for the relevance rankings

        :return: relevance ranking based on DVS
        '''
        if len(self.collected_documents) > 0:
            return solr.search('{!knn f=bertbase topK=10}' + str(self.embedding),**{'rows':10, 'fq':self.prefilter()})
        else:
            return solr.search('{!knn f=bertbase topK=10}' + str(self.embedding), **{'rows':10})

    def iterate(self) -> None:
        '''
        iterate resembles an iteration in the review/feedback process (for 10 documents)

        :return: None
        '''
        results = self.do_query()
        user_feedback = []
        for result in results:
            if args.Topic in result['topic']:
                self.collected_documents.add(result['docId'])
                user_feedback.append(result['text'][0])
            else:
                self.declined_documents.add(result['docId'])
        self.keywords += process_feedback(user_feedback)

    def recall(self) -> float:
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
