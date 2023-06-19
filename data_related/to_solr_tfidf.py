import pandas as pd, ast
import pysolr

solr = pysolr.Solr('http://localhost:8983/solr/tfidf')
data = []

def load_reuters() -> pd.DataFrame:
    '''
    load_reuters reads the csv file that contains RCV-1 v2

    :return: pandas dataframe
    '''
    return pd.read_csv('docs/ambigious.csv', sep=';')

if __name__ == '__main__':
    df = load_reuters()

    for index, row in df.iterrows(): 
        topic = [x for x in ast.literal_eval(row["topic"])]
        data.append({"topic":topic, "text":row["text"], "id":index}) # "docId":row['docId'],
        if index % 200 == 0:
            print(index)
            solr.add(data, commit=True)
            solr.commit()
            solr.ping()
            data = []

    solr.add(data, commit=True)
    solr.commit()