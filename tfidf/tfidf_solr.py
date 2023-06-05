__file__ = 'tfidf_solr.py'
__author__ = 'Timo Kats'
__description__ = 'Conducts the experiment for TFIDF using Solr MoreLikeThis.'

# libraries

import pysolr, argparse, json
import numpy as np, pandas as pd
from random import random

solr = pysolr.Solr('http://localhost:8983/solr/tfidf')

cutoffs = {
    10: {"TP":0,"FP":0,"FN":0},
    20: {"TP":0,"FP":0,"FN":0},
    50: {"TP":0,"FP":0,"FN":0},
    100: {"TP":0,"FP":0,"FN":0},
    200: {"TP":0,"FP":0,"FN":0},
    300: {"TP":0,"FP":0,"FN":0},
    500: {"TP":0,"FP":0,"FN":0}
    }

# initialize parser

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--Topic", help = "Topic of the source document")
args = parser.parse_args()

# I/O functions

def output_results(m,precision,recall,f1_score):
    results = {
        "topic":args.Topic,
        "top k":round(m,4),
        "precision":round(precision,5),
        "recall":round(recall,5),
        "f1_score": round(f1_score,5)
    }
    filename = 'results/test.json'
    with open(filename,'a+', encoding='utf-8') as f:
        json.dump(results, f)
        f.write('\n')
    print(results)

# solr functions

def get_data(query, no_docs) -> tuple:
    results = solr.search(query, **{'rows':no_docs})
    for index, result in enumerate(results):
        update_results(result['topic'], index)

# tfidf (MoreLikeThis)

def format_query(query) -> str:
    return "({!mlt qf=text fl=topic}" + query + " AND topic:*)"

# scoring

def get_scores(results) -> tuple:
    recall = float(results["TP"] / (results["TP"] + results["FN"]))
    precision = float(results["TP"] / (results["TP"] + results["FP"]))
    f1_score = float((2*precision*recall)/(precision + recall))
    return round(precision,4), round(recall,4), round(f1_score,4)

def update_results(topics, current_index):
    global cutoffs
    for cutoff, result in cutoffs.items():
        if cutoff >= current_index and args.Topic in topics:
            cutoffs[cutoff]["TP"] += 1
        elif cutoff >= current_index  and args.Topic not in topics:
            cutoffs[cutoff]["FP"] += 1
        elif cutoff < current_index and args.Topic in topics:
            cutoffs[cutoff]["FN"] += 1

# get queries

def get_queries():
    query_data = {"topic":[],"id":[], "text":[]}
    queries = solr.search('topic:(' + args.Topic + ')', **{'rows':300})
    for query in queries:
        query_data['topic'].append(query['topic'])
        query_data['id'].append(query['id'])
        query_data['text'].append(query['text'][0])
    return query_data

# main

if __name__ == '__main__':
    args = parser.parse_args()
    queries = get_queries()

    for query, text, topic in zip(queries['id'], queries['text'], queries['topic']):
        tfidf_query = format_query(query)
        get_data(tfidf_query, 7500)

    for cutoff, result in cutoffs.items():
        precision,recall,f1_score = get_scores(result)
        output_results(cutoff,precision,recall,f1_score)
