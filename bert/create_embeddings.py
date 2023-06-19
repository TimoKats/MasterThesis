__file__ = 'bert_solr.py'
__author__ = 'Timo Kats'
__description__ = 'Creates S-BERT embeddings for the BERT experiments.'

# libraries

from sentence_transformers import SentenceTransformer
import pandas as pd, numpy as np

def get_data() -> pd.DataFrame:
    '''
    load_data reads the csv file that contains RCV-1 v2

    :return: pandas dataframe
    '''
    return pd.read_csv('paragraphs/test_paragraphs.csv', sep=';')

def output_embeddings(embeddings) -> None:
    '''
    output_embeddings outputs the SBERT embeddings to a .npy file

    :param embeddings: Numpy array with SBERT embeddings
    :return: None
    '''
    filename = 'embeddings/mini/test_3.npy'
    with open(filename, 'wb') as file:
        np.save(file, embeddings)
    file.close()

if __name__ == '__main__':
    data = get_data()
    text = list(data['text'])
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embedding = model.encode(text)
    output_embeddings(embedding)
    
