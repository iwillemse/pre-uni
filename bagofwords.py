import numpy as np

import json
import heapq

#get texts
with open('10k_tokenized_texts.json', 'r') as file:
    tokenized_texts = json.load(file)

#count word frequency and create vocabulary
wordfreq = {}
for tokenized_text in tokenized_texts:
    for token in tokenized_text:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1

#get 10k most frequent words
most_freq = heapq.nlargest(10000, wordfreq, key=wordfreq.get)

#create bag of words vectors
bow_vecs = []
for i in range(len(tokenized_texts)):
    bow_vec = [0] * len(most_freq)
    for j in range(len(most_freq)):
        occurence = tokenized_texts[i].count(most_freq[j])/(len(tokenized_texts[i])+1)
        bow_vec[j] = occurence
    bow_vecs.append(bow_vec)

#dump to files
with open('10k_bow_embeds.json', 'w') as file:
    json.dump(bow_vecs, file)
