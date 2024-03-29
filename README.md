# Using text similarity and relevance feedback to reduce review effort in eDiscovery
This repositiory contains the source code of Timo Kats' master thesis: *Using text similarity and relevance feedback to reduce review effort in eDiscovery*. This thesis was written for the ICT in Business master program at the Leiden University and was part of a Data Science internship at ZyLAB. Supervisors: Ludovic Jean-Louis, Zoe Gerolemou, Jan Scholtes and Peter van der Putten.  

### Overview
This repository consists of 6 folders: bert, tfidf, quorum, relevance_feedback, data_related and PDF. The first three of these folders refer to the text similarity experiments. The TF-IDF and BERT (in the thesis also referred to as dense vector search) based experiments require Solr to be installed. For this, please see the section **Solr**. Note, a tutorial on setting up the DVS/relevance feedback experiments can be found [here](https://www.loom.com/share/47951dacf8ca4c64994d34add5fdb198?sid=bdf1436c-4e4e-4ab3-b8a1-5851b7644851)  

Finally, the manuscript and defence of the thesis can be found in the folder PDF.

### System information and requirements
The experiment was run on a Windows 11 enviroment using the Windows Subsystem for Linux. The Python version for this was *Python 3.8.10*. Moreover, the experiments use a collection of libraries. These are listed (along with their version number) in `requirements.txt` (see command for installation below). Finally, most experiments require Solr (at least version 9) to be installed. For this, please see section **Solr**.  

```python
pip3 install -r requirements.txt
```

Some of the paths referred to in the source code are related to the machine the experiments were originally run on. Thus, these might not correlate with the layout of your device. Hence, feel free to change these filepaths on your local instance.  

### Data
The experiment uses the RCV-1 v2 dataset for the experiments. This dataset is under the copyright of Reuters Ltd and/or Thomson Reuters. Hence, its contents are not published in this repository. However, if you have this dataset then you can load its (XML based) contents using the script `data_related/read_reuters.py`, which will return a json file. The contents of this file can then be used to push the data to Solr using `to_solr_bert.py` and `to_solr_tfidf.py` after setting up Solr. More information on setting up Solr can be found in the next section. 

### Solr
Most experiment use Solr as a search engine, which can be downloaded and installed for free at https://solr.apache.org/. Note, for the dense vector search, a version >9 needs to be installed. The version used in this research is Solr 9.2.0.  

After installing, a Solr instance needs to be setup for the experiments. For the TF-IDF based experiments, the schema for this is added in XML format at `data_related/schemas/tfidf_schema.xml`. For the BERT/DVS based experiments and relevance feedback experiments, the schema for this is added in XML format at `data_related/schemas/bert_schema.xml`. 

### Help/Comments
Every function in the source code is provided with PEP 8 compliant commentary. As a result, you can call a help function for each function to learn more about its role. An example for this is given below. Moreover, all the function are provided with type hints.

```
>>> from quorum.results import *
>>> help(get_scores)

Help on function get_scores in module quorum:

get_scores(results) -> tuple
    get_scores takes the TP, FN, and FPs and returns precision/recall/f1 scores

    :param results: dictionary with TPs, FPs and FNs
    :return: tuple with recall, precision and f1 score

>>>
```

### Run the experiments
The experiments are often run automatically using shell scripts. For this, a number of command line arguments are used. Hereby an overview of these command line arguments. These can be appended to calling the Python script.

| argument code | meaning                                        | applicable to                                                                                                                    |
|---------------|------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| -n            | Refers to the recall of the Quorum operator    | quorum.py                                                                                                                        |
| -m            | Refers to the precision of the Quorum operator | quorum.py                                                                                                                        |
| -t            | Refers to the topic used in the experiment     | tfidf_solr.py,    tfidf_svm.py,   bert_solr.py,    quorum.py,   vector_feedback.py,  rocchio_feedback.py,  keyword_feedback.py,  |
| -s            | Refers to the usage of stopwords (True/False)  | create_embeddings.py                                                                                                             |


### Recommended readings
 - https://sease.io/2023/01/apache-solr-neural-search-tutorial.html 
