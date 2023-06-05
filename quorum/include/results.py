__file__ = 'include/results.py'
__author__ = 'Timo Kats'
__description__ = 'Functions related to processing the Quorum returned files and getting results.'

# globals

cutoffs = {
    10: {"TP":0,"FP":0,"FN":0},
    20: {"TP":0,"FP":0,"FN":0},
    50: {"TP":0,"FP":0,"FN":0},
    100: {"TP":0,"FP":0,"FN":0},
    200: {"TP":0,"FP":0,"FN":0},
    300: {"TP":0,"FP":0,"FN":0},
    500: {"TP":0,"FP":0,"FN":0}
}

def update_results_cutoff(index, same_topic):
    for cutoff, result in cutoffs.items():
        if index <= cutoff and same_topic:
            cutoffs[cutoff]["TP"] += 1
        elif index <= cutoff and not same_topic:
            cutoffs[cutoff]["FP"] += 1
        elif index > cutoff and same_topic:
            cutoffs[cutoff]["FN"] += 1

def update_results(similarities, topic_documents):
    for index, (document, matches) in enumerate(similarities.items()):
        if document in topic_documents:
            update_results_cutoff(index, True)
        else:
            update_results_cutoff(index, False)

def sort_similarities(similarities, doc_lengths, m):
    sorted_similarities = {}
    for doc, matches in similarities.items():
        m_int = int(float(m)*doc_lengths[doc])
        if matches >= m_int:
            sorted_similarities[doc] = matches/doc_lengths[doc]
    return {k: v for k, v in sorted(sorted_similarities.items(), key=lambda item: item[1], reverse=True)}

def get_scores(results) -> tuple:
    recall = float(results["TP"] / (results["TP"] + results["FN"]))
    precision = float(results["TP"] / (results["TP"] + results["FP"]))
    f1_score = float((2*precision*recall)/(precision + recall))
    return round(recall,4), round(precision,4), round(f1_score,4)

def output_results(cutoff,topic,precision,recall,f1_score):
    results = {
        "Topic":topic,
        "Cutoff@":round(cutoff,3),
        "Precision":round(precision,5),
        "Recall":round(recall,5),
        "F1 score": round(f1_score,5)
    }
    filename = 'test.json'
    #with open(filename,'a+', encoding='utf-8') as f:
    #    json.dump(results, f)
    #    f.write('\n')
    print(results)
