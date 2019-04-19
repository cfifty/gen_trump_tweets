import nltk
import itertools

import pandas as pd
import numpy as np 


vocab_size = 16000
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
	vocab = word_freq.most_common(vocab_size)

	index_to_word = {(index+1):word[0] for index,word in enumerate(vocab)}
	index_to_word[vocab_size] = unknown_token

	word_to_index = {word[0]:(index+1) for index,word in enumerate(vocab)}
	word_to_index[unknown_token] = vocab_size

	unknown_lookup = [x for x in word_freq if x not in word_to_index]

	# process original tweets to replace rare words with the unknown token
	for index,tweet in enumerate(tokenized_tweets):
		tokenized_tweets[index] = [twit if twit in word_to_index else unknown_token for twit in tweet]

	# map words to integers for training data
	X_train = np.asarray([[word_to_index[twit] for twit in tweet[:-1]] for tweet in tokenized_tweets])
	Y_train = np.asarray([[word_to_index[twit] for twit in tweet[1:]] for tweet in tokenized_tweets])

	return X_train,Y_train, index_to_word, word_to_index, vocab_size, unknown_lookup

def gen_tweets(sess, num_tweets, top_n, index_to_word, unknown_lookup, num_layers, batch_size, num_hidden_units):
	for _ in range(num_tweets):
		sentence = []
		counter = 0
		next_token = np.ones((128,1)) # start token 
		next_LSTM_state = np.zeros((num_layers, 2, batch_size, num_hidden_units))
		
		# while an end token hasn't been generated
		while(next_token[0] != 2): 
			gen_word = index_to_word[next_token[0][0]]
			if gen_word == unknown_token:
				gen_word = unknown_lookup[np.random.randint(len(unknown_lookup))]
			sentence.append(gen_word)

			preds, next_LSTM_state = sess.run([logits_final, H], feed_dict={X_batch:next_token, LSTM_state:next_LSTM_state})

			# sample from probabilities
			p = np.squeeze(preds[0]) # get the first row... can delete when you fix variable batch size...
			p[np.argsort(p)][:-top_n] = 0 # set the first n - top_n indices to 0
			p = p/np.sum(p)
			index = np.random.choice(vocab_size, 1, p=p)[0]

			next_token = np.ones((128,1))*index
			counter += 1

			if counter > 20: # let's say tweets can't be > 20 words...
				break

		sentence = sentence[1:] # get rid of the <s> token
		print(" ".join(sentence))