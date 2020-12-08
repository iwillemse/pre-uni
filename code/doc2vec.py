import numpy as np
import pandas as pd
import json

import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.keyedvectors import KeyedVectors

#get tokenized texts
with open('10k_tokenized_texts.json', 'r') as file:
    tokenized_texts = json.load(file)

#tag documents and train doc2vec model
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokenized_texts)]
model = Doc2Vec(documents, vector_size=300, window=5, min_count=20, workers=4, dm=0)
#for PV-DM use dm=1 instead of dm=0, default is PV-DBOW

#embed all the texts with trained doc2vec model
embedded_texts = [np.ndarray] * len(tokenized_texts)
for i in range(len(tokenized_texts)):
    embedded_texts[i] = model.infer_vector(tokenized_texts[i])

#convert from np.array to float
for i in range(len(embedded_texts)):
    embedded_texts[i] = list(embedded_texts[i])
    for j in range(len(embedded_texts[i])):
        embedded_texts[i][j] = embedded_texts[i][j].item()

#dump to files
with open('10k_doc2vec_dbow_embeds.json', 'w') as file:
    json.dump(embedded_texts, file)
