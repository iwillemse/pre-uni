import numpy as np
import pandas as pd
import json
import heapq

# K-means
from sklearn.cluster import KMeans

# Plotting
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

# TSNE
from sklearn.manifold import TSNE
import seaborn as sns
sns.set_style('white')
sns.set_palette("muted")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

# load embedded texts and convert to np array
with open('10k_bow_embeds.json', 'r') as file:
    embedded_texts = json.load(file)
X = np.array(embedded_texts)

#load tokenized texts and titles
with open('10k_str_titles.json', 'r') as file:
    titles = json.load(file)
with open('10k_tokenized_texts.json', 'r') as file:
    tokenized_texts = json.load(file)

#find k and plot sse-k graph
sse = []
ran = range(1, 31)
for k in ran:
    res = KMeans(init="k-means++", n_clusters=(k)).fit(X)
    sse.append(res.inertia_)
    print(k)
plt.style.use("default")
plt.plot(ran, sse)
plt.show()

#run k-means with found k
knee = 11
    
kmeans = KMeans(init="k-means++", n_clusters=knee)
res = kmeans.fit(X)
labs = kmeans.labels_

#visualise clustering with t-SNE
tsne = TSNE(n_components=2, random_state=0)
tsne_obj = tsne.fit_transform(X)

tsne_embedded_texts = pd.DataFrame({'X':tsne_obj[:,0], 'Y':tsne_obj[:,1], 'digit':labs})

fig, ax = plt.subplots(1, 1, figsize = (7, 7), dpi=150)
sns_plot = sns.scatterplot(x="X", y="Y", hue=labs, palette=sns.color_palette("Paired", knee),legend="full", data=tsne_embedded_texts).get_figure()
ax.set_ylabel('')    
ax.set_xlabel('')

#sort texts and titles by cluster
labs = labs.tolist()
titles_clusters = [['0'], ['1'], ['2'], ['3'], ['4'], ['5'], ['6'], ['7'], ['8'], ['9'], ['10']]
tokenized_texts_clusters = [['0'], ['1'], ['2'], ['3'], ['4'], ['5'], ['6'], ['7'], ['8'], ['9'], ['10']]
for i in range(len(labs)):
    titles_clusters[labs[i]].append(titles[i])
    tokenized_texts_clusters[labs[i]].append(tokenized_texts[i])

#get 25 most frequent words per cluster
most_freq = []
for i in range(11):
    wordfreq = {}
    for tokenized_text in tokenized_texts_clusters[i]:
        for token in tokenized_text:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1
    most_freq.append(heapq.nlargest(25, wordfreq, key=wordfreq.get))

#dump to files
with open('10k_bow_kmeans_clustered_titles.json', 'w') as file:
    json.dump(titles_clusters, file)
with open('10k_doc2vec_dbow_most_freq.json', 'w') as file:
    json.dump(most_freq, file)
