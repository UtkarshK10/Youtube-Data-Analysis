from textblob import TextBlob
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from kneed import KneeLocator

df=pd.read_csv("USvideos.csv")
df['description'].head()
df['title'].head()

def process_text(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity
'''sentiment analysis for description of youtube video.'''
df['descriptionthought'] = df.apply(lambda row: process_text(str(row.description)), axis = 1)
df['descriptionSentiment']=np.where(df['descriptionthought']<0.0,0,1)
df['descriptionSentiment'].head() 
df.head()

'''sentimental analysis of title of youtube videos.'''
df['titlethought'] = df.apply(lambda row: process_text(str(row.title)), axis = 1)
df['titleSentiment']=np.where(df['titlethought']<0.0,0,1)
df['titleSentiment'].head() 
df.head()

'''
   reason behind getting the sentiments of both title and description 
   is to make sure that any user who is showing positivity in its title 
   but not doing any fraud in the description segment to show negativity
   in name of positivity.
'''

'''convertion dataframe into csv file so that we can use it as input for hadoop'''
compression_opts = dict(method='zip',archive_name='youtube.csv')  
df.to_csv('youtube.zip', index=False,compression=compression_opts) 

''' kMeans clustering using different Attributes'''
df=pd.read_csv("youtube.csv")

'''clustering on basis of likes and comment_count. and plotting the graph'''
X=df[['likes', 'comment_count']].values

''' Feature Scaling '''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X=sc_X.fit_transform(X)


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
plt.show()

'''optimal value of k we get here by using elbow method is 3 '''
kmeans =KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = '+ve content')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'unpredictable/neutral')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'pink', label = '-ve content')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of youtube data')
plt.xlabel('views')
plt.ylabel('title Sentiment')
plt.legend()
plt.show()

'''
   Reason for the plot of title_sentiment and description_sentiment.
__main__:5: ConvergenceWarning: Number of distinct clusters (4) found smaller than n_clusters (5). Possibly due to duplicate points in X.
__main__:5: ConvergenceWarning: Number of distinct clusters (4) found smaller than n_clusters (6). Possibly due to duplicate points in X.
__main__:5: ConvergenceWarning: Number of distinct clusters (4) found smaller than n_clusters (7). Possibly due to duplicate points in X.
__main__:5: ConvergenceWarning: Number of distinct clusters (4) found smaller than n_clusters (8). Possibly due to duplicate points in X.
__main__:5: ConvergenceWarning: Number of distinct clusters (4) found smaller than n_clusters (9). Possibly due to duplicate points in X.
__main__:5: ConvergenceWarning: Number of distinct clusters (4) found smaller than n_clusters (10). Possibly due to duplicate points in X.
'''



''' using pricipal component analysis for unsupervised learning model to reduce the dimensions.'''
X=df[['views','likes','dislikes','comment_count','descriptionSentiment','titleSentiment']].values

''' Feature Scaling '''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X=sc_X.fit_transform(X)

from sklearn.decomposition import PCA
pca=PCA(n_components=None)
X=pca.fit_transform(X)
explained_variance=pca.explained_variance_ratio_


X=df[['views', 'likes']].values

''' Feature Scaling '''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X=sc_X.fit_transform(X)


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
plt.show()

'''optimal value of k we get here by using elbow method is 3 '''
kmeans =KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'less popular content')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'people\'s choice')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'pink', label = 'most popular content')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of youtube data')
plt.xlabel('views')
plt.ylabel('likes')
plt.legend()
plt.show()



import warnings
from collections import Counter
import datetime
import wordcloud
import json

title_words = list(df["title"].apply(lambda x: x.split()))
title_words = [x for y in title_words for x in y]
Counter(title_words).most_common(25)

wc = wordcloud.WordCloud(width=1200, height=500,collocations=False, background_color="white",colormap="tab20b").generate(" ".join(title_words))
plt.figure(figsize=(15,10))
plt.imshow(wc, interpolation='bilinear')
_ = plt.axis("off")

'''
def elbow(X):
    wcss=[]
    for i in range(1,11):
        kmeans=KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1,11),wcss, 'bx-',)
    plt.title('the elbow method')
    plt.xlabel('number of clusters')
    plt.ylabel('WCSS')
    x = range(1, len(wcss)+1)
    kn = KneeLocator(x, wcss, curve='convex', direction='decreasing')
    plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.show()
elbow(X)
'''