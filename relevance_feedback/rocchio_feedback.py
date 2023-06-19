__file__ = 'rocchio_feedback.py'
__author__ = 'Timo Kats'
__description__ = 'Conducts relevance feedback experiments for rocchio.'

# libraries

import pysolr, argparse, numpy as np, pandas as pd
from itertools import chain

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
    print(result)
    filename = 'results/rocchio/test/test_rocchio.csv'
    with open(filename, 'a+', encoding='utf-8') as f:
        f.write(str(round(np.mean(result),3)) + ';' + str(round(np.std(result),3)) + '\n')

# solr

def get_embeddings() -> list:
    '''
    get_embeddings returns the SBERT embeddings of the queries

    :return: list with embeddings (either first or random paragraphs)
    '''
    query_data = {"docId":[], "embedding":[]}
    queries = solr.search('topic:(' + args.Topic + ')', **{'rows':5000})
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

# fancy experiments

class RocchioFeedback:
    def __init__(self, embedding, alpha, beta):
        self.collected_documents = set()
        self.declined_documents = set()

        self.original_embedding = embedding
        self.current_embedding = embedding
        self.feedback_embeddings = []

        self.size = get_maxsize()
        self.iterations = 1
        self.alpha = alpha
        self.beta = beta

    # vector functions

    def average_vectors(self, feedback) -> list:
        '''
        average_vectors returns the average of a set of vectors

        :param feedback: list of vectors that resemble SBERT embeddings
        :return: integer with amount of documents
        '''
        feedback.append(self.current_embedding)
        return list(np.mean(feedback, axis=0))

    def sum_vectors(self, feedback) -> list:
        '''
        sum_vectors returns the sum of a set of vectors

        :param feedback: list of vectors that resemble SBERT embeddings
        :return: integer with amount of documents
        '''
        feedback.append(self.current_embedding)
        return list(np.add.reduce(feedback))

    def amplify_feedback(self, docId) -> None:
        '''
        amplify_feedback adds embeddings from sibling paragraphs to the feedback

        :param docId: document identifier of the parent document
        :return: integer with amount of documents
        '''
        results = solr.search('docId:' + docId)
        for result in results:
            self.feedback_embeddings.append(result['bertbase'])

    def rocchio_vectors(self) -> None:
        '''
        rochhio_vectors updates the queried vector akin Rocchio's formula

        :return: None
        '''
        self.current_embedding = [self.alpha * i + self.beta * j for i, j in zip(self.original_embedding, self.average_vectors(self.feedback_embeddings))]

    # query functions

    def prefilter(self) -> str:
        '''
        prefilter applies the filter created in filter_docs

        :return: string that contains filter based on approved/declined documents formatted for Solr
        '''
        filter = ""
        for docId in chain(self.collected_documents, self.declined_documents):
            filter += ' ' + str(docId) + ' '
        return 'NOT docId: (' + filter + ')'

    def do_query(self) -> pysolr.Results:
        '''
        do_query executes the DVS for the relevance rankings

        :return: relevance ranking based on DVS
        '''
        if len(self.collected_documents) > 0:
            return solr.search('{!knn f=bertbase topK=10}' + str(self.current_embedding),**{'rows':10, 'fq':self.prefilter()})
        else:
            return solr.search('{!knn f=bertbase topK=10}' + str(self.current_embedding), **{'rows':10})

    def iterate(self) -> None:
        '''
        iterate resembles an iteration in the review/feedback process (for 10 documents)

        :return: None
        '''
        docIds = set()
        results = self.do_query()
        for result in results:
            if args.Topic in result['topic']:
                self.feedback_embeddings.append(result['bertbase'])
                self.collected_documents.add(result['docId'])
                docIds.add(result['docId'])
            else:
                self.declined_documents.add(result['docId'])
        self.rocchio_vectors()

    def recall(self):
        return len(self.collected_documents) / self.size

# main

if __name__ == '__main__':
    result = []
    for embedding in get_embeddings():
        vector_session = RocchioFeedback(embedding, 0.5, 0.5)
        while vector_session.recall() < 0.8:
            vector_session.iterate()
            vector_session.iterations += 1
        result.append(vector_session.iterations)
    export(result)
