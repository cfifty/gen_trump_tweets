import utils 

import numpy as np
import tensorflow as tf 

# must match what was saved
batch_size = 128
num_hidden_units = 200
num_layers = 3

num_tweets = 50
max_tweet_len = 20
top_n = 20

X_train, Y_train, index_to_word, word_to_index, vocab_size, unknown_lookup = utils.load_dataset()
vocab_size += 1 # due to 0 being used as padding...

ending_punc = ['.', ',', '?', '!', '"', "'", ":", '...']
starting_punc = ['"', '“']
never_cap = ['http']

def postprocess(sentence_lst):
	sentence_acc = ""
	prev_word = ""
	for word in sentence_lst:

		# basic grammar handling
		if (prev_word in ['.', '?', '!'] or prev_word in [""]) and ord(word[0]) >= ord('a') and ord(word[0]) <= ord('z') and 'http' not in word:
			word = word.capitalize()

		# basic spacing handling....
		if len(sentence_acc) == 0:
			sentence_acc += word
		elif prev_word in starting_punc and (len(sentence_acc) <= 1 or sentence_acc[-1] != " "):
			sentence_acc += word
		elif word not in ending_punc or prev_word in ['"', "'"]:
			sentence_acc += " {}".format(word)
		else:
			sentence_acc += word

		prev_word = word
	return sentence_acc



with tf.Session() as sess: 
	loader = tf.train.import_meta_graph('checkpoints/trump_lstm-740.meta')
	loader.restore(sess, 'checkpoints/trump_lstm-740')
	
	while(num_tweets):
		sentence = []
		counter = 0
		next_token = np.ones((128,1)) # start token 
		next_LSTM_state = np.zeros((num_layers, 2, batch_size, num_hidden_units))
		
		# while an end token hasn't been generated
		while(next_token[0] != 2): 
			gen_word = index_to_word[next_token[0][0]]
			if gen_word == utils.unknown_token:
				gen_word = unknown_lookup[np.random.randint(len(unknown_lookup))]
			sentence.append(gen_word)

			preds, next_LSTM_state = sess.run(['logits_final:0', 'H:0'], feed_dict={'Placeholder:0':next_token, 'Placeholder_2:0':next_LSTM_state})

			# sample from probabilities
			p = np.squeeze(preds[0]) # get the first row... can delete when you fix variable batch size...
			p[np.argsort(p)][:-top_n] = 0 # set the first n - top_n indices to 0
			p = p/np.sum(p)
			index = np.random.choice(vocab_size, 1, p=p)[0]

			next_token = np.ones((128,1))*index
			counter += 1

			if counter > max_tweet_len: # let's say tweets can't be > 20 words...
				break

		if counter < max_tweet_len:
			num_tweets -= 1

			sentence = sentence[1:] # get rid of the <s> token
			print(postprocess(sentence))
			#print(" ".join(sentence))