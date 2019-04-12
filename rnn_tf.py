import os

import numpy as np
import tensorflow as tf

from preprocess import load_dataset

X_train, Y_train, index_to_word, word_to_index, vocab_size, unknown_lookup = load_dataset()
vocab_size += 1 # due to using 0 as a padding...

# define neural net constants 
num_epochs = 500
batch_size = 128
num_hidden_units = 200
num_layers = 3
embedding_dim = 300
num_batches = (X_train.shape[0])//batch_size
lr = 0.01
save_rate = 10 
unknown_token = "<unk>"

# define placeholders: input, output, and rnn hidden state
X_batch = tf.placeholder(tf.int32, shape=(batch_size, None))							# shape: (batch size x max_seq_len)
X_one_hot = tf.one_hot(X_batch, vocab_size, on_value=1, off_value=0)					# convert to one-hot encoding

Y_batch = tf.placeholder(tf.int32, shape=(batch_size, None))
Y_one_hot = tf.one_hot(Y_batch, vocab_size, on_value=1, off_value=0) 					# shape: (batch size x max_seq_len x vocab size = depth)

# determine the length of this batch's max sequence -- for per-example sequence shortening -- don't update based on paddings
length = tf.cast(tf.reduce_sum(tf.sign(X_batch), 1), tf.int32)

# add an embedding layer
word_embeddings = tf.get_variable(name="word_embeddings", shape=(vocab_size, embedding_dim))
embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, X_batch)					# shape: (batch size x max_seq_len x embedd size)

# define the LSTM parameters (i.e. initializers)
LSTM_state = tf.placeholder(tf.float32, shape=(num_layers, 2, batch_size, num_hidden_units))
LSTM_state_lst = tf.unstack(LSTM_state,axis=0)
LSTM_tuple_lst = tuple([tf.nn.rnn_cell.LSTMStateTuple(LSTM_state_lst[i][0], LSTM_state_lst[i][1]) for i in range(num_layers)])

# define the forward pass
cells = []
for _ in range(num_layers):
	cell = tf.nn.rnn_cell.LSTMCell(num_hidden_units, state_is_tuple=True)
	cells.append(cell)

cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
outputs, H = tf.nn.dynamic_rnn(cell, embedded_word_ids, initial_state=LSTM_tuple_lst, sequence_length=length)	# shape of outputs: batch_size x max_seq_length x num_hidden_units =z> rather than a |max_seq_length| list of batch_size x vocab length tensors 
H = tf.identity(H, name="H")

# mask over the time steps in each example to see which ones are padding
mask = tf.sign(tf.reduce_max(tf.abs(outputs), 2))	# (batch_size x max_seq_len): value = 0 <=> it is padding

# last layer is fully connected to go to vocab_size
outputs = tf.reshape(outputs, shape=(-1, num_hidden_units))			# output from LSTM: batch_size*max_seq_length * hidden unit size
logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)
logits = tf.reshape(logits, shape=(batch_size, -1, vocab_size)) # expand back out to (batch_size x max_seq_len x vocab_size)
logits_final = tf.nn.softmax(logits, name='logits_final')

# commented out for custom cross-entropy cost function
# labels = tf.reshape(Y_one_hot, shape=(-1, vocab_size))				# labels: batch_size*max_seq_length x vocab size
# labels = tf.cast(labels, tf.float32) 
# losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
# total_loss = tf.reduce_mean(losses)

# compute the loss -- special since we're considering variable sequence lengths
labels = tf.cast(Y_one_hot, tf.float32)
cross_entropy = labels * tf.log(logits_final)
cross_entropy = -tf.reduce_sum(cross_entropy,axis=2) # sum the cross entropy along the vocab_size axis into a single value
cross_entropy *= mask # eliminate the cross_entropy values from time steps that are padding
cross_entropy = tf.reduce_sum(cross_entropy, axis=1) # sum cross entropy along the time_steps axis
cross_entropy /= tf.reduce_sum(mask,axis=1) # collapse the mask into (batch_size, ) and sum it to get the sequence length for each example -- divide each sequence by its length to normalize it
cross_entropy = tf.reduce_mean(cross_entropy)

total_loss = cross_entropy
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)

saver = tf.train.Saver(max_to_keep=3)

init = tf.global_variables_initializer()

# train 
with tf.Session() as sess:
	sess.run(init)
	loss_list = []

	for epoch_idx in range(num_epochs):
		# shuffle the dataset
		random_indices = np.random.permutation(X_train.shape[0])
		X_train = X_train[random_indices]
		Y_train = Y_train[random_indices]

		print("epoch: {}".format(epoch_idx))
		init_LSTM_state = np.zeros((num_layers, 2, batch_size, num_hidden_units))

		for batch_idx in range(num_batches):
			start_idx = batch_idx*batch_size
			end_idx = batch_idx*batch_size + batch_size

			X = X_train[start_idx:end_idx]
			Y = Y_train[start_idx:end_idx]

			# variable minibatch sequence lengths
			batch_seq_len = 0
			for row in range(X.shape[0]):
				batch_seq_len = max(batch_seq_len, len(X[row]))
			
			X_mat = np.zeros((batch_size, batch_seq_len))
			Y_mat = np.zeros((batch_size, batch_seq_len))
			for row in range(batch_size):
				for col in range(len(X[row])):
					X_mat[row][col] += X[row][col]
					Y_mat[row][col] += Y[row][col]


			# print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) # prints out all trainable variables in the dataflow graph :O 
			# print("first example mask: {}".format(sess.run(mask, feed_dict={X_batch:X_mat, Y_batch:Y_mat, LSTM_state:init_LSTM_state})[0,:]))
			# print("biases added to each index in vocab: {}".format(sess.run("fully_connected/biases:0", feed_dict={X_batch:X_mat, Y_batch:Y_mat, LSTM_state:init_LSTM_state})))

			_, minibatch_cost = sess.run([optimizer, total_loss], feed_dict={
																				X_batch:X_mat, 
																				Y_batch:Y_mat, 
																				LSTM_state:init_LSTM_state,
																			})

			loss_list.append(minibatch_cost)

		print("epoch loss: {}".format(np.average(loss_list)))
		if epoch_idx % save_rate == 0:
			saver.save(sess, 'checkpoints/trump_lstm', global_step=epoch_idx)
	# put in inference mode
	for _ in range(200):
		sentence = []
		counter = 0
		next_token = np.ones((128,1)) # start token
		next_LSTM_state = np.zeros((num_layers, 2, batch_size, num_hidden_units))

		# while an end token hasn't been generated...
		while(next_token[0] != 2): 
			gen_word = index_to_word[next_token[0][0]]
			if gen_word == unknown_token:
				gen_word = unknown_lookup[np.random.randint(len(unknown_lookup))]
			sentence.append(gen_word)

			preds, next_LSTM_state = sess.run([logits_final, H], feed_dict={X_batch:next_token, LSTM_state:next_LSTM_state})

			# sample from probabilities
			p = np.squeeze(preds[0])
			p = p/np.sum(p)
			index = np.random.choice(vocab_size, 1, p=p)[0]

			next_token = np.ones((128,1))*index
			counter += 1

			if counter > 20:
				print("counter is greater than 20... breaking")
				break

		sentence = sentence[1:] # get rid of the <s> token
		print(" ".join(sentence))



