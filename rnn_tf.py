import numpy as np
import tensorflow as tf

from preprocess import load_dataset

X_train, Y_train, index_to_word, word_to_index, vocab_size = load_dataset()

# define neural net constants 
num_epochs = 10
batch_size = 128
num_hidden_units = 50
num_layers = 3
max_seq_len = 73
embedding_dim = 300
backprop_truncate = 10
num_batches = (X_train.shape[0])//batch_size

# define placeholders: input, output, and rnn hidden state
X_batch = tf.placeholder(tf.int32, shape=(batch_size, max_seq_len))
X_one_hot = tf.one_hot(X_batch, vocab_size, on_value=1, off_value=0)					# convert to one-hot encoding
assert(X_one_hot.shape == (batch_size, max_seq_len, vocab_size))

Y_batch = tf.placeholder(tf.int32, shape=(batch_size, max_seq_len))
Y_one_hot = tf.one_hot(Y_batch, vocab_size, on_value=1, off_value=0) 					# shape: (batch size x max_seq_len x vocab size = depth)
assert(Y_one_hot.shape == (batch_size, max_seq_len, vocab_size))

# add an embedding layer
word_embeddings = tf.get_variable(name="word_embeddings", shape=(vocab_size, embedding_dim))
embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, X_batch)					# shape: (batch size x max_seq_len x embedd size)
assert(embedded_word_ids.shape == (batch_size, max_seq_len, embedding_dim))

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
outputs, _ = tf.nn.dynamic_rnn(cell, embedded_word_ids, initial_state=LSTM_tuple_lst)	# shape of outputs: batch_size x max_seq_length x num_hidden_units =z> rather than a |max_seq_length| list of batch_size x vocab length tensors 
outputs = tf.reshape(outputs, shape=(-1, num_hidden_units))			# output from LSTM: batch_size*max_seq_length * hidden unit size

# last layer is fully connected to go to vocab_size
logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)
labels = tf.reshape(Y_one_hot, shape=(-1, vocab_size))				# labels: batch_size*max_seq_length x vocab size

# compute the loss
losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
total_loss = tf.reduce_mean(losses)

optimizer = tf.train.AdagradOptimizer(0.3).minimize(total_loss)
init = tf.global_variables_initializer()

# train 
with tf.Session() as sess:
	sess.run(init)
	loss_list = []

	for epoch_idx in range(num_epochs):
		print("epoch: {}".format(epoch_idx))
		init_LSTM_state = np.zeros((num_layers, 2, batch_size, num_hidden_units))

		for batch_idx in range(num_batches):
			start_idx = batch_idx*batch_size
			end_idx = batch_idx*batch_size + batch_size

			X = X_train[start_idx:end_idx,:]
			Y = Y_train[start_idx:end_idx,:]

			_, minibatch_cost = sess.run([optimizer, total_loss], feed_dict={
																				X_batch:X, 
																				Y_batch:Y, 
																				LSTM_state:init_LSTM_state,
																			})

			loss_list.append(minibatch_cost)

		print("epoch loss: {}".format(np.average(loss_list)))








