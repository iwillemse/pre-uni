import numpy as np # linear algebra
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import re
import os

#get texts and titles from pdf.json files in kaggle environment
texts = []
titles = []
filepaths = os.listdir('../input/CORD-19-research-challenge/document_parses/pdf_json/')
n=0
m=0

for filepath in filepaths:
    filepath_full = os.path.join('../input/CORD-19-research-challenge/document_parses/pdf_json/', filepath)
    if filepath_full.endswith(".json"):
        with open(filepath_full) as f:
            data = json.load(f)
        doc = ''
        title = data['metadata']['title']
        if title == '': title = 'title_blank' #some titles in the json are blank, changes these to 'title_blank' to avoid confusion
        title = title.replace('\n', '').replace('\t', '') #removes newlines and tabs from titles
        for i in range(len(data['body_text'])):
            doc += data['body_text'][i]['text']
        if(doc != ''):
            texts.append(doc)
            titles.append(title)
        n+=1
        m+=1
    if m >= 1000:
        print(n)
        m = 0
    if n >= 10000:
        break

#preprocess text in string form
for i in range(len(texts)):
    texts[i] = texts[i].lower() #makes lowercase
    texts[i] = re.sub('[^A-Za-z ]+', '', texts[i]) #removes punctuation and numbers
    texts[i] = re.sub(' +', ' ', texts[i]) #removes extra whitespaces
    texts[i] = texts[i].replace('\n', '').replace('\t', '') #removes newlines and tabs


extra_stopwords = ['et', 'al', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

#tokenize text and remove stopwords
tokenized_texts = [[]] * len(texts)
stop_words = set(stopwords.words('english') + extra_stopwords)
for i in range(len(texts)):
    tokens = word_tokenize(texts[i])
    tokenized_texts[i] = [j for j in tokens if not j in stop_words]

#remake strings (without stopwords) from tokens
for i in range(len(texts)):
    texts[i] = ('').join(tokenized_texts[i])

#dump to files
with open('10k_tokenized_texts.json', 'w') as file:
    json.dump(tokenized_texts, file)
with open('10k_str_texts.json', 'w') as file:
    json.dump(texts, file)
with open('10k_str_titles.json', 'w') as file:
    json.dump(titles, file)
