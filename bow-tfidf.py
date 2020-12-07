import json
import heapq
import math

#get texts
with open('10k_tokenized_texts.json', 'r') as file:
    tokenized_texts = json.load(file)

#count word frequency and create vocabulary
wordfreq = {}
for text in tokenized_texts:
    for token in text:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1

#get 10k most frequent words
import heapq
most_freq = heapq.nlargest(10000, wordfreq, key=wordfreq.get)

#count document occurence (= in how many different documents a word appears)
document_occurence = [0] * len(most_freq)
for i in range(len(most_freq)):
    for text in tokenized_texts:
        if most_freq[i] in text:
            document_occurence[i] += 1

#get inverse document frequency (idf) for each word
idf = [0] * len(most_freq)
for i in range(len(most_freq)):
        idf[i] = (math.log(len(tokenized_texts)/document_occurence[i]))

#create bag of words vectors with tf-idf weighting
tfidf_vecs = []
for i in range(len(tokenized_texts)):
    tfidf_vec = [0] * len(most_freq)
    for j in range(len(most_freq)):
        tf = tokenized_texts[i].count(most_freq[j])/len(tokenized_texts[i]) #weighs document length
        tfidf_vec[j] = tf * idf[j]
    tfidf_vecs.append(tfidf_vec)

#dump to files
with open('10k_bow_tfidf_embeds.json', 'w') as file:
    json.dump(tfidf_vecs, file)
