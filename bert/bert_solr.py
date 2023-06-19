__file__ = 'bert_solr.py'
__author__ = 'Timo Kats'
__description__ = 'Conducts the experiment for BERT using dense vector search.'

# libraries

import pysolr, pandas as pd, argparse
from random import random

# solr

solr = pysolr.Solr('http://localhost:8983/solr/ms-marco')

# initialize parser

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--Topic", help = "Topic with the source documents")
args = parser.parse_args()

# globals

scores = {    
    10: {"TP":0,"FP":0,"FN":0},
    20: {"TP":0,"FP":0,"FN":0},
    50: {"TP":0,"FP":0,"FN":0},
    100: {"TP":0,"FP":0,"FN":0},
    200: {"TP":0,"FP":0,"FN":0},
    300: {"TP":0,"FP":0,"FN":0},
    500: {"TP":0,"FP":0,"FN":0}
    }

def update_scores(returned_documents) -> None:
    '''
    update_scores counts TPs FPs and FNs based on a relevance ranking

    :param returned_documents: relevance ranking in a dictionary format
    :return: None
    '''
    for index, (docId, topic) in enumerate(returned_documents.items()):
        for threshold in scores.keys():
            if index <= threshold and args.Topic in topic:
                scores[threshold]['TP'] += 1
            elif index <= threshold and args.Topic not in topic:
                scores[threshold]['FP'] += 1
            elif index > threshold and args.Topic in topic:
                scores[threshold]['FN'] += 1

def update_scores_parcount(returned_documents, count_docIds) -> None:
    '''
    update_scores counts TPs FPs and FNs based on a relevance ranking (specified to paragraph count)

    :param returned_documents: relevance ranking in a dictionary format
    :param count_docIds: Matches per document in a dictionary format
    :return: None
    '''
    for index, docId in enumerate(count_docIds.items()):
        for threshold in scores.keys():
            if index <= threshold and args.Topic in returned_documents[docId]:
                scores[threshold]['TP'] += 1
            elif index <= threshold and args.Topic not in returned_documents[docId]:
                scores[threshold]['FP'] += 1
            elif index > threshold and args.Topic in returned_documents[docId]:
                scores[threshold]['FN'] += 1

def get_documents(results) -> tuple:
    '''
    get_documents derives document ranking from returned paragraphs

    :param results: relevance ranking in paragraphs
    :return: Tuple of dictionaries that contian the document level paragraph rankings
    '''
    count_docIds = {}
    first_docIds = []
    returned_documents = {}
    for result in results:
        returned_documents[result['docId']] = result['topic']
        if result['docId'] not in count_docIds.keys(): # count approach
            count_docIds[result['docId']] = 1
        else:
            count_docIds[result['docId']] += 1
        if result['docId'] not in first_docIds: # first approach
            first_docIds.append(result['docId'])
    return returned_documents, {k: v for k, v in sorted(count_docIds.items(), key=lambda item: item[1], reverse=True)}

def export_results() -> None:
    '''
    export_results gets the results and writes it to csv

    :return: None
    '''
    f = open('results/tfidf/random-mlt.csv', 'a+')
    for threshold, score in scores.items():
        recall = float(score["TP"] / (score["TP"] + score["FN"]))
        precision = float(score["TP"] / (score["TP"] + score["FP"]))
        f1_score = float((2*precision*recall)/(precision + recall))
        print(precision, recall, f1_score)
        f.write(args.Topic + ';' + str(threshold) + ';' + str(round(recall,4)) + ';' + str(round(precision,4))  + ';' + str(round(f1_score,4)) + '\n')

def get_queries() -> pd.DataFrame:
    '''
    get_queries returns all queries for an experiment given a topic

    :return: Dictionary of lists with query data (select first or random paragraph)
    '''
    query_data = {"docId":[],"topic":[],"embedding":[], "id":[]}
    queries = solr.search('topic:(' + args.Topic + ')', **{'rows':7500})
    for query in queries:
        query_data['docId'].append(query['docId'])
        query_data['topic'].append(query['topic'])
        query_data['embedding'].append(query['bertbase'])
        query_data['id'].append(query['id'])
    query_df = pd.DataFrame.from_dict(query_data)
    return query_df.groupby('docId').sample(n=1).reset_index(drop=True) # .head(1) for first paragraph

if __name__ == '__main__':
    args = parser.parse_args()
    queries = get_queries()
    for index, row in queries.iterrows():
            query = "{!knn f=bertbase topK=7500}" + str(row['embedding']) 
            results = solr.search(query, **{'rows':7500, 'fl':'topic,docId,text'})
            returned_documents, count_docIds = get_documents(results)
            update_scores(returned_documents) # or update_scores_parcount()
    export_results()
