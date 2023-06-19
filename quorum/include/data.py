__file__ = 'include/data.py'
__author__ = 'Timo Kats'
__description__ = 'Various functions that load or export data.'

# libraries

import pandas as pd

def get_positive_set(data, topic):
    '''
    get_positive_set gets the documents with the same topic

    :param topic: string that has the selected topic name
    :return: list with indecies of the positive set
    '''
    indicies= []
    for index, row in data.iterrows():
        if topic in row['topic']:
            indicies.append(index)
    return indicies

def load_reuters(filename):
    '''
    load_reuters reads the csv file that contains RCV-1 v2

    :param filename: path to the filename
    :return: pandas dataframe
    '''
    return pd.read_csv(filename, encoding='utf-8', sep=';')
