__file__ = 'tfidf_svm.py'
__author__ = 'Timo Kats'
__description__ = 'Runs Linear-SVM experiment. Requires the TFIDF embeddings made in create_embeddings.py'

# libraries

import json, pandas as pd, numpy as np, argparse, ast
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# initialize parser

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--Topic", help = "Topic with the source documents")
args = parser.parse_args()

# globals

clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))

# functions

def load_topics(filepath) -> list:
    result = []
    topics = list(pd.read_csv(filepath, sep=';')['topic'])
    for index, topic in enumerate(topics):
        if args.Topic in ast.literal_eval(topic):
            result.append(True)
        else:
            result.append(False)
    return result

def load_embedding(filepath) -> np.ndarray:
    return np.load(filepath)

# SVM

def score_classifier(embeddings, topics) -> dict:
    results = {'TP':0,'FP':0,'FN':0}
    for embedding, topic in zip(embeddings, topics):
        prediction = clf.predict([embedding])[0]
        if prediction and topic:
            results['TP'] += 1
        elif prediction and not topic:
            results['FP'] += 1
        elif not prediction and topic:
            results['FN'] += 1
    return results

def get_scores(results) -> tuple:
    recall = float(results["TP"] / (results["TP"] + results["FN"]))
    precision = float(results["TP"] / (results["TP"] + results["FP"]))
    f1_score = float((2*precision*recall)/(precision + recall))
    return round(recall,4), round(precision,4), round(f1_score,4)

def export_scores(recall, precision, f1_score):
    f = open('svm_results/ambigious_linear_svm.csv', 'a+')
    f.write(args.Topic + ';' + str(recall) + ';' + str(precision) + ';' + str(f1_score) + '\n')
    print(recall, precision, f1_score)

if __name__ == '__main__':
    args = parser.parse_args()
    train_topics = load_topics('../data/svm/svm_train.csv')
    test_topics = load_topics('../data/svm/svm_test.csv')

    train_embeddings = load_embedding('embeddings/svm.npy')[:len(train_topics)]
    test_embeddings = load_embedding('embeddings/svm.npy')[-len(test_topics):]

    clf.fit(train_embeddings, train_topics)

    results = score_classifier(test_embeddings, test_topics)
    recall, precision, f1_score = get_scores(results)
    export_scores(recall, precision, f1_score)

