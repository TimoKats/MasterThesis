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
    '''
    update_results_cutoff adds results to the global result counter

    :param index: the positition in the relevance ranking
    :param same_topic: bool that captures the topic of the query and result
    :return: None
    '''
    for cutoff, result in cutoffs.items():
        if index <= cutoff and same_topic:
            cutoffs[cutoff]["TP"] += 1
        elif index <= cutoff and not same_topic:
            cutoffs[cutoff]["FP"] += 1
        elif index > cutoff and same_topic:
            cutoffs[cutoff]["FN"] += 1

def update_results(similarities, topic_documents):
    '''
    update_results iterates through the relevance ranking and updats the results

    :param similarities: dictionary with the relevance ranking
    :param topic_documents: list of indecies that have the same topic as the query
    :return: None
    '''
    for index, (document, matches) in enumerate(similarities.items()):
        if document in topic_documents:
            update_results_cutoff(index, True)
        else:
            update_results_cutoff(index, False)

def sort_similarities(similarities, doc_lengths, m):
    '''
    sort_similarities takes the relevance ranking and re-sorts it akin to Quorum search

    :param similarities: dictionary with the (original) relevance ranking
    :param doc_lengths: dictionary with document lengths
    :param m: threshold (expressed as a percentage of document length)
    :return: None
    '''
    sorted_similarities = {}
    for doc, matches in similarities.items():
        m_int = int(float(m)*doc_lengths[doc])
        if matches >= m_int:
            sorted_similarities[doc] = matches/doc_lengths[doc]
    return {k: v for k, v in sorted(sorted_similarities.items(), key=lambda item: item[1], reverse=True)}

def get_scores(results) -> tuple:
    '''
    get_scores takes the TP, FN, and FPs and returns precision/recall/f1 scores

    :param results: dictionary with TPs, FPs and FNs
    :return: Tuple with recall, precision and f1 score
    '''
    recall = float(results["TP"] / (results["TP"] + results["FN"]))
    precision = float(results["TP"] / (results["TP"] + results["FP"]))
    f1_score = float((2*precision*recall)/(precision + recall))
    return round(recall,4), round(precision,4), round(f1_score,4)

def output_results(cutoff,topic,precision,recall,f1_score):
    '''
    output_results gets the results and writes it to json

    :param cutoff: int with current cutoff in the relevance ranking
    :param topic: name of the selected topic
    :param precision: float with precision score
    :param recall: float with recall score
    :param f1_score: float with f1 score
    :return: None
    '''
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
