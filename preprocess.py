import numpy as np
import pandas as pd
import numpy as np 
import tensorflow as tf 
from sklearn.feature_extraction.text import CountVectorizer 


corpus = pd.read_csv('data/trump_tweets.csv', low_memory=False).text.tolist()
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.vocabulary_)
print(len(vectorizer.vocabulary_))