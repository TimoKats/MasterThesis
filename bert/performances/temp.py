from sentence_transformers import SentenceTransformer
import pandas as pd
import psutil, time

def get_data():
    return pd.read_csv('paragraphs/test_paragraphs.csv', sep=';')

if __name__ == '__main__':
    data = get_data()
    text = list(data['text'])
    model = SentenceTransformer('all-MiniLM-L6-v2')
    f = open('L6performance.csv', 'a+')

    for paragraph in text:
        time_start = time.time()
        model.encode(paragraph)
        time_end = time.time()
        f.write(str(psutil.virtual_memory().percent) + ';' + str(psutil.cpu_percent()) + ';' + str(round(time_end-time_start,4)) + '\n')
