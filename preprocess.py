import nltk
import itertools

import pandas as pd
import numpy as np 


vocab_size = 4000
unknown_token = "<unk>"
start_token = "<s>"
end_token = "</s>"

def load_dataset():
	longest_seq = 0

	tweet_lst = pd.read_csv('data/trump_tweets.csv', low_memory=False).text.tolist()
	tokenized_tweets = []
	for tweet in tweet_lst:
		tokenized_tweet = nltk.tokenize.TweetTokenizer().tokenize(tweet.lower())
		tokenized_tweet = [start_token] + tokenized_tweet + [end_token]
		longest_seq = max(longest_seq, len(tokenized_tweet))
		tokenized_tweets.append(tokenized_tweet)

	print("Your longest sequence has length: {}".format(longest_seq))

	# build a vocabulary from the most common words
	word_freq = nltk.FreqDist((itertools.chain(*tokenized_tweets)))
	vocab = word_freq.most_common(vocab_size-1)

	index_to_word = {index:word[0] for index,word in enumerate(vocab)}
	index_to_word[vocab_size-1] = unknown_token

	word_to_index = {word[0]:index for index,word in enumerate(vocab)}
	word_to_index[unknown_token] = vocab_size-1

	# process original tweets to replace rare words with the unknown token
	for index,tweet in enumerate(tokenized_tweets):
		tokenized_tweets[index] = [twit if twit in word_to_index else unknown_token for twit in tweet]

	# map words to integers for training data
	X_train = np.asarray([[word_to_index[twit] for twit in tweet[:-1]] for tweet in tokenized_tweets])
	Y_train = np.asarray([[word_to_index[twit] for twit in tweet[1:]] for tweet in tokenized_tweets])

	X = np.zeros((X_train.shape[0], longest_seq))
	for row in range(X_train.shape[0]):
		for col in range(len(X_train[row])):
			X[row][col] = X_train[row][col]

	Y = np.zeros((Y_train.shape[0], longest_seq))
	for row in range(Y_train.shape[0]):
		for col in range(len(Y_train[row])):
			Y[row][col] = Y_train[row][col]

	X_train = X
	Y_train = Y 

	print("X_train shape: {}".format(X_train.shape))
	print("Y_train.shape: {}".format(Y_train.shape))
	print(X_train[0])

	return X_train,Y_train, index_to_word, word_to_index, vocab_size

# load_dataset()