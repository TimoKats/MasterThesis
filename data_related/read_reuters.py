__file__ = 'read_reuters.py'
__author__ = 'Timo Kats'
__description__ = 'Reads the reuters RCV1 v2 disks and converts it to a jsonl file. requires both disks.'

import os, json
from bs4 import BeautifulSoup 


def get_topic_mappings() -> dict:
    '''
    get_topic_mappings (optional) converts the topic codes to topic names

    :return: dictionary of mappings
    '''
    topic_mapping = {}
    with open('topic_list.txt', 'r') as topics:
        for topic in topics:
            code = topic.split('\t')[0]
            name = topic.split('\t')[1][:-1]
            topic_mapping[code] = name
    return topic_mapping

def get_topic(data) -> list:
    '''
    get_topic gets the topics of an RCV1 xml instance

    :return: list of topics
    '''
    cleaned_topics = []
    try:
        topics = data.find('codes', {'class':'bip:topics:1.0'}).findAll('code')
        for topic in topics:
            cleaned_topics.append(topic.attrs['code'])
    except:
        topics = None
    return cleaned_topics

def get_text(data) -> list:
    '''
    get_text gets the text of an RCV1 xml instance

    :return: list of paragraphs (often sentences) with text
    '''
    paragraph_text = []
    paragraphs = data.find('text').findAll('p')
    for paragraph in paragraphs:
        paragraph_text.append(paragraph.text)
    return paragraph_text

def get_headline(data) -> str:
    '''
    get_headline gets the title of an RCV1 xml instance

    :return: string with the title
    '''
    return data.find('headline').text

def get_id(data) -> str:
    '''
    get_id gets the unique article identifier of an RCV1 xml instance

    :return: list of paragraphs (often sentences) with text
    '''
    return data.find('newsitem').attrs['itemid']

def get_date(data) -> str:
    '''
    get_date gets the date published of an RCV1 xml instance

    :return: list of paragraphs (often sentences) with text
    '''
    return data.find('newsitem').attrs['date']  

def output_json(data) -> None:
    '''
    output_json appends the collected RCV1 instance to a json file

    :return: None
    '''
    with open("rcv_ambigious_paragraphs.jsonl", "a+") as outfile:
        for par in data["text"]:
            output = {"docId":data["docId"],"topic":data["topic"],"text":par}
            print(output)
            json.dump(output, outfile)
            outfile.write('\n')
            outfile.flush()

if __name__ == '__main__':
    output = {"docId":0, "topic": [], "text": []}

    for filename in os.scandir('disk1'):
        if filename.is_file():
            with open(filename.path, 'r', encoding='utf-8') as f:
                try:
                    data = f.read() 
                    data = BeautifulSoup(data, 'lxml') 
                    output["topic"] = get_topic(data)
                    output["text"] = get_text(data)
                    output_json(output)
                    output["docId"] += 1
                except:
                    pass

    for filename in os.scandir('disk2'):
        if filename.is_file():
            with open(filename.path, 'r', encoding='utf-8') as f:
                try:
                    data = f.read() 
                    data = BeautifulSoup(data, 'lxml') 
                    output["topic"] = get_topic(data)
                    output["text"] = get_text(data)
                    output_json(output)
                    output["docId"] += 1
                except:
                    pass
