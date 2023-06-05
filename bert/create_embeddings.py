__file__ = 'bert_solr.py'
__author__ = 'Timo Kats'
__credits__ = ['Ludovic Jean-Louis', 'Zoe Gerolemou', 'Johannes Scholtes']
__description__ = 'Creates S-BERT embeddings for the BERT experiments.'

# libraries

from sentence_transformers import SentenceTransformer
import pandas as pd, numpy as np

def get_data():
    return pd.read_csv('paragraphs/test_paragraphs.csv', sep=';')

def output_embeddings(embeddings):
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
    
