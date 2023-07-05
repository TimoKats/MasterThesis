import pandas as pd, ast
import pysolr, numpy as np

solr = pysolr.Solr('http://localhost:8983/solr/ms-marco')
data = []
whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')

def load_reuters():
    '''
    load_reuters reads the csv file that contains RCV-1 v2

    :return: pandas dataframe
    '''
    return pd.read_csv('paragraphs/test_paragraphs.csv', sep=';')

def load_embeddings(size):
    '''
    load_embeddings loads numpy file with SBERT embeddings

    :size: refers to the type of pre-trained model used {small, medium, base}
    :return: pandas dataframe
    '''
    with open('embeddings/' + size + '/test_3.npy', 'rb') as file:
        embeddings = np.load(file)
        return embeddings

if __name__ == '__main__':
    df = load_reuters()
    #l6_embeddings = load_embeddings('mini')
    #l12_embeddings = load_embeddings('medium')
    base_embeddings = load_embeddings('base')

    for index, row in df.iterrows():
        print(index)
        #l6_embedding = [float(w) for w in list(l6_embeddings[index])] 
        #l12_embedding = [float(w) for w in list(l12_embeddings[index])] 
        base_embedding = [float(w) for w in list(base_embeddings[index])]
        topic = [x for x in ast.literal_eval(row["topic"])]

        data.append({"id": index, "topic":topic, "text":row["text"], "docId":row["docId"], "bertbase":base_embedding})
        if index % 200 == 0:
            solr.add(data, commit=True)
            solr.commit()
            solr.ping()
            data = []

    solr.add(data, commit=True)
    solr.commit()

