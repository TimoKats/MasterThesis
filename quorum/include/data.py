__file__ = 'include/data.py'
__author__ = 'Timo Kats'
__description__ = 'Various functions that load or export data.'

# libraries

import pandas as pd

def get_positive_set(data, topic):
    indicies= []
    for index, row in data.iterrows():
        if topic in row['topic']:
            indicies.append(index)
    return indicies

def load_reuters(filename):
    return pd.read_csv(filename, encoding='utf-8', sep=';')
