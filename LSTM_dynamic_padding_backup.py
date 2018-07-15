import tensorflow as tf
import numpy as np
import pickle
from pprint import pprint
from sklearn.model_selection import train_test_split



# Fake data params
max_class_id = 5 # y_true = activity
max_n_frame = 50 # max_n_frames
max_freq = 3 # obj_freq_in_frame
n_feature = 33 # bag-of-objects
n_video = 20

# NN params
lstm_in_cell_units=5 # design choice (hyperparameter)

# training params
n_epoch = 5
batch_size=4
learning_rate=0.01
# ********************************************************
#!!!!IMPORTANTEEEEE!!!
# handling last batch remainder
n_iteration = n_video//batch_size
# *********************************************************



#---------------DATASET----------------

videos_detection = []

for i in range(n_video):
	current_class_id = np.random.randint(0,max_class_id)
	current_n_frame = np.random.randint(1,max_n_frame+1)
	#current_n_frame = 40

	current_sequence=[]
	for j in range(current_n_frame):
		current_frame = np.random.randint(0,high=max_freq,size=(1,n_feature), dtype=np.uint8).tolist()[0]
		current_sequence.append(current_frame)

	videos_detection.append({'sequence':current_sequence, 
							 'activity_id':current_class_id,
							 'n_frame':current_n_frame})


pickle.dump(videos_detection, open('videos_detection.pickle', 'wb'))



videos_detection = pickle.load(open('videos_detection.pickle', 'rb'))




#*********************************
# TODO: DATA PREPROCESSING STEPS
#*********************************


X,y,seq_len=[],[],[]
# dataset input, output, seq_len construction
for index,i in enumerate(videos_detection):
	X.append([frame_detection for frame_detection in i['sequence']])
	one_hot = [0]*max_class_id
	one_hot[i['activity_id']-1] = 1
	y.append(one_hot)
	seq_len.append(i['n_frame'])


X_train, X_test, y_train, y_test, seq_len_train, seq_len_test = \
	 train_test_split(X,y,seq_len,test_size=0.3, random_state=0, stratify=y)





#---------------GRAPH------------------

# zip input output and sequence_length
zipped_data = list(zip(X_train,y_train,seq_len_train))
# dataset
data = tf.data.Dataset.from_generator(lambda: zipped_data, (tf.int32, tf.int32, tf.int32))
# shuffle the (whole) dataset
data = data.shuffle(buffer_size=len(X))

# obtain a padded_batch and push it with an iterator
shape = ([None,n_feature],[max_class_id],[])
data_batch = data.padded_batch(batch_size, padded_shapes=shape)

iterator = data_batch.make_initializable_iterator()
current_batch = iterator.get_next()

# slice the dataset in X, y, seq_len
current_X_batch = tf.cast(current_batch[0], dtype=tf.float32)
current_y_batch = current_batch[1]
current_seq_len_batch = tf.reshape(current_batch[2], (1,-1))[0]


# lstm
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_in_cell_units, state_is_tuple=True)
state_c, state_h = lstm_cell.zero_state(batch_size, tf.float32)
initial_states = tf.nn.rnn_cell.LSTMStateTuple(tf.Variable(state_c, trainable=False), 
											   tf.Variable(state_h, trainable=False))

outputs, states = tf.nn.dynamic_rnn(lstm_cell, 
								   current_X_batch, 
								   initial_state=initial_states,
								   sequence_length=current_seq_len_batch, 
								   dtype=tf.float32)

# last_step_output done right (each instance will have it's own seq_len therefore the right last ouptut for each instance must be taken)
last_step_output = tf.gather_nd(outputs, tf.stack([tf.range(tf.shape(current_X_batch)[0]), current_seq_len_batch-1], axis=1))

# logits
logits = tf.layers.dense(last_step_output, units=max_class_id)

# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=current_y_batch))

# optimization
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))


init = tf.global_variables_initializer()

# session
with tf.Session() as sess:
	sess.run(init)

	for i in range(n_epoch):
		print('\nEpoch: %d' % (i+1))
		sess.run(iterator.initializer)
		for j in range(n_iteration):
			print('Batch: %d' % (j+1))
			# addesso l'eccezione non verrà lanciata perchè salto 
			# l'ultimo piccolo batch sempre 
			# per gestire il resto nel caso in cui n_video e batch_size non fosser multipli
			# lo devo fare perchè la dimensione dello zero_state dipende (deve essere uguale)
			# a batch_size e quindi nel caso di n_video e batch_size non multpili
			# per l'ultimo batch succede un casino perchè la sua size non è uguale a batch_size
			try:
				# results = sess.run((outputs,
				# 					last_step_output,
				# 					tf.stack([tf.range(tf.shape(current_X_batch)[0]), current_seq_len_batch-1], axis=1),
				# 					current_X_batch, 
				# 					current_y_batch, 
				# 					current_seq_len_batch,
				# 					current_batch))
				# pprint(results)
				_, current_loss = sess.run((optimizer, loss))					
				print('Loss: %f' % current_loss)
			except tf.errors.OutOfRangeError:
				print('EOF')
				break





#TODO
# + shuffle the batch
#	double check the class distro of the batch!!!! It must be similar to the whole dataset class distro!!!!
# + handle padding loss mask

# rifletti su dynamic vs static e sul fatto del padding 
# tu l'hai fatto basandoti sulla sequenza che ha la lunghezza massima
# TRA TUTTE QUELLE PRESENTI NEL DATASET e non TRA TUTTE QUELLE ALL'INTERNO DI UN BATCH
# SEE GERON TEXTBOOK


# + tanh default inner LSTM state activation (known to be the best for LSTMs)
# check batch normalization ???
# GLOROT/HE weights initialization

# add accuracy op (look at the link)
# take a look at the graph
# add the Tensorboard loss function handler
# add the evaluation mode
# 
# tune the hyperparameters?
