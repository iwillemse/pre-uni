import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import json

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from sklearn.manifold import TSNE
import seaborn as sns
sns.set_style('ticks')
sns.set_palette("muted")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

#load embedded texts and convert to np array
with open('10k_doc2vec_dbow_embeds.json', 'r') as file:
    embedded_texts = json.load(file)
X = np.array(embedded_texts)

#DBSCAN
tX = X

pca = PCA(n_components=100) # test verschillende n_components voor pca
principalComponents = pca.fit_transform(X)
tX = principalComponents

neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(tX)
distances, indices = nbrs.kneighbors(tX)

distances = np.sort(distances, axis=0)
distances = distances[:,1]

plt.style.use("default")
plt.plot(distances) # gebruiken om de elleboog (=eps) te vinden
plt.show()
plt.savefig('./elbow.png')

res = DBSCAN(eps=10, min_samples=10).fit(tX)
knee = len(set(list(res.labels_)))
labs = list(res.labels_)

# Visualise clustering
tsne = TSNE(n_components=2, random_state=0)
tsne_obj = tsne.fit_transform(X)
tsne_embedded_texts = pd.DataFrame({'X':tsne_obj[:,0], 'Y':tsne_obj[:,1], 'digit':labs})

fig, ax = plt.subplots(1, 1, figsize = (7, 7), dpi=150)
sns_plot = sns.scatterplot(x="X", y="Y", hue=labs, palette=sns.color_palette("Paired", knee),legend='full', data=tsne_embedded_texts).get_figure()
ax.set_ylabel('')    
ax.set_xlabel('')

sns_plot.savefig('./tsne.png')
