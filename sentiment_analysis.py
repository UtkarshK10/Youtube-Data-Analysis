import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

'''nlp_libraries'''
import string
import re
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize


df = pd.read_csv("youtube.csv")
eng_stopwords = set(stopwords.words("english"))

'''Word count in each comment'''
df['word']=df["title"].apply(lambda x: len(str(x).split()))
df['word_tags']=df["tags"].apply(lambda x: len(str(x).split()))

'''Unique word count'''
df['unique_word']=df["title"].apply(lambda x: len(set(str(x).split())))
df['unique_word_tags']=df["tags"].apply(lambda x: len(set(str(x).split())))

'''Letter count'''
df['letters']=df["title"].apply(lambda x: len(str(x)))
df['letters_tags']=df["tags"].apply(lambda x: len(str(x)))

'''punctuation count'''
df["punctuations"] =df["title"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
df["punctuations_tags"] =df["tags"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

'''upper case words count'''
df["upper"] = df["title"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
df["upper_tags"] = df["tags"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

'''title case words count'''
df["words_title_uppercase"] = df["title"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
df["words_tags_uppercase"] = df["tags"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

'''Number of stopwords'''
df["stopwords"] = df["title"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
df["stopwords_tags"] = df["tags"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

'''Average length of the words'''
df["mean_len"] = df["title"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
df["mean_len_tags"] = df["tags"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))


df['Normalizedlikes'] = np.log(df['likes'] + 1)
df['Normalizedviews'] = np.log(df['views'] + 1)
df['Normalizeddislikes'] = np.log(df['dislikes'] + 1)
df['Normalizedcomment'] = np.log(df['comment_count'] + 1)

'''punctuations in title'''
plt.figure(figsize = (12,8))
plt.subplot(221)
g=sns.boxplot(x='punctuations', y='Normalizedviews',data=df)
g.set_title("Ponctuation wise views")
g.set_xlabel("Numer of Punctuations")
g.set_ylabel("Vews")

plt.subplot(222)
g1 = sns.boxplot(x='punctuations', y='Normalizedlikes',data=df)
g1.set_title("Punctuation wise likes")
g1.set_xlabel("Numer of Punctuations")
g1.set_ylabel("Likes")

plt.subplot(223)
g2 = sns.boxplot(x='punctuations', y='Normalizeddislikes',data=df)
g2.set_title("Ponctuation wise dislikes")
g2.set_xlabel("Numer of Punctuations")
g2.set_ylabel("Dislikes")

plt.subplot(224)
g3 = sns.boxplot(x='punctuations', y='Normalizedcomment',data=df)
g3.set_title("Ponctuation wise comments")
g3.set_xlabel("Numer of Punctuations")
g3.set_ylabel("Comments")

plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)
plt.show()


'''tags punctuation'''
plt.figure(figsize = (12,8))

plt.subplot(221)
g=sns.boxplot(x='punctuations_tags', y='Normalizedviews',data=df[df['punctuations_tags'] < 10])
g.set_title("Punctuations in tags wise views")
g.set_xlabel("Numer of Tag Punctuations")
g.set_ylabel("Vews")

plt.subplot(222)
g1 = sns.boxplot(x='punctuations_tags', y='Normalizedlikes',data=df[df['punctuations_tags'] < 10])
g1.set_title("Punctuations in tags wise likes")
g1.set_xlabel("Numer of Tag Punctuations")
g1.set_ylabel("Likes")

plt.subplot(223)
g2 = sns.boxplot(x='punctuations_tags', y='Normalizeddislikes',data=df[df['punctuations_tags'] < 10])
g2.set_title("Punctuations in tags wise dislikes")
g2.set_xlabel("Numer of Tag Punctuations")
g2.set_ylabel("Dislikes")

plt.subplot(224)
g3 = sns.boxplot(x='punctuations_tags', y='Normalizedcomment',data=df[df['punctuations_tags'] < 10])
g3.set_title("Punctuations in tags wise comment")
g3.set_xlabel("Numer of Tag Punctuations")
g3.set_ylabel("Comments")

plt.subplots_adjust(wspace = 0.2, hspace = 0.4,top = 0.9)
plt.show()

'''correlation matrix'''
plt.figure(figsize = (12,8))
sns.heatmap(df[['word', 'unique_word','letters',
                     "punctuations","upper", "words_title_uppercase", 
                     "stopwords","mean_len", 
                     'Normalizedviews', 'Normalizedlikes','Normalizeddislikes','Normalizedcomment',
                     'ratings_disabled', 'comments_disabled', 'video_error_or_removed']].corr(), annot=True)
plt.show()