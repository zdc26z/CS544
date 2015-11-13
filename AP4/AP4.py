#Kholby Lawson, CS544, Fall 2015
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from collections import Counter

'''

#  Attribute
-- -------------------------------------------
1. area A, 
2. perimeter P, 
3. compactness C = 4*pi*A/P^2, 
4. length of kernel, 
5. width of kernel, 
6. asymmetry coefficient 
7. length of kernel groove. 
All of these parameters were real-valued continuous.

'''

#read raw data
names=['area', 'perimeter', 'compactness', 'length', 'width', 'asymmetry coefficient', 'groove', 'type']
raw_data = pd.read_table('seeds_dataset.txt', delimiter='\t', names=names)
data = raw_data[names[0:len(names)-1]]
variety = raw_data['type']

k = 3

kmeans = KMeans(n_clusters=k)
clusters = kmeans.fit_predict(data)
cvar = [[] for _ in range(k)]
total = 0

for i in range(0,k):
    c = np.where(clusters==i)[0].tolist()
    cv = variety[c]
    count = Counter(cv).most_common(1)
    cvar[i] = count[0][0]
    total = total + count[0][1]

print 'Accuracy:  ', float(total)/len(variety)
centers = kmeans.cluster_centers_.tolist()

print '\nCentroids:\n'

for i in range(0,k):
    centers[i].append(cvar[i])
    print zip(names,centers[i])
    






