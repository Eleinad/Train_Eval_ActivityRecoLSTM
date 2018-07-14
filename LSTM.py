import tensorflow as tf
import numpy as np
import pickle
from pprint import pprint



# Fake data params
max_class_id = 5 # y_true = activity
max_n_frame = 5 # max_n_frames
max_freq = 3 # obj_freq_in_frame
n_feature = 4 # bag-of-objects
n_video = 10


# NN params
batch_size=5
lstm_in_cell_units=5 # design choice (hyperparameter)
learning_rate=0.01
n_epoch = 3




#---------------DATASET----------------

videos_detection = []

for i in range(n_video):
	current_class_id = np.random.randint(0,max_class_id)
	current_n_frame = np.random.randint(1,max_n_frame+1)
	#current_n_frame = 40

	current_sequence=[]
	for j in range(current_n_frame):
		current_frame = np.random.randint(0,high=max_freq,size=(1,n_feature), dtype=np.uint8)
		current_sequence.append(current_frame)

	videos_detection.append({'sequence':current_sequence, 
							 'activity_id':current_class_id,
							 'n_frame':current_n_frame})


pickle.dump(videos_detection, open('videos_detection.pickle', 'wb'))



#pprint(videos_detection)

videos_detection = pickle.load(open('videos_detection.pickle', 'rb'))


#*********************************
# TODO: DATA PREPROCESSING STEPS
#*********************************


# get the max n_frame for padding
# all videos with n_frame (sequence) < max_frame will be right zero padded
# so that all sequences are the same length
max_frame = max([i['n_frame'] for i in videos_detection])

# dataset matrices construction 
X = np.zeros((len(videos_detection), max_frame*n_feature), dtype=np.uint8)
y = np.zeros((len(videos_detection), max_class_id), dtype=np.uint8)
seq_len = np.zeros((len(videos_detection), 1), dtype=np.int32)

# dataset matrices filling -> unfilled zeros = right padding
for index,i in enumerate(videos_detection):
	j=0
	for frame_detection in i['sequence']:
		for obj_freq in frame_detection[0]:
			X[index, j] = obj_freq
			j+=1
	y[index, i['activity_id']] = 1
	seq_len[index] = i['n_frame']




#---------------TENSORFLOW------------------

# dataset
data = tf.data.Dataset.from_tensor_slices((X, y, seq_len))

# shuffle the (whole) dataset
data = data.shuffle(buffer_size=X.shape[0])

# obtain a batch and push it with iterator
data_batch = data.batch(batch_size)
iterator = data_batch.make_initializable_iterator()
current_batch = iterator.get_next()

# slice the dataset in X, y, seq_len
current_X_batch = current_batch[0]
current_y_batch = current_batch[1]
current_seq_len_batch = tf.reshape(current_batch[2], (1,-1))[0]

current_X_batch = tf.cast(current_X_batch, dtype=tf.float32)

# split per static_rnn --> crea una lista di lunghezza t
# con Tensor di shape (n_instances_in_current_batch, n_feature)
# Ex.: time = 2 --> len(seq) = 2 = max_frame therefore [X0, X1]
# where X0.shape = X1.shape = (n_instances_in_current_batch, n_features)
current_X_sequence_batch = tf.split(current_X_batch, num_or_size_splits=max_frame, axis=1)

# lstm
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_in_cell_units)#, forget_bias=1.0, state_is_tuple=True)
outputs, states = tf.nn.static_rnn(lstm_cell, 
								   current_X_sequence_batch, 
								   sequence_length=current_seq_len_batch, 
								   dtype=tf.float32)

#last_step_output done right (each instance will have it's own seq_len therefore the right last ouptut for each instance must be taken)
last_step_output = tf.gather_nd(outputs, tf.stack([current_seq_len_batch-1, tf.range(tf.shape(current_X_batch)[0])], axis=1))

# logits
logits = tf.layers.dense(last_step_output, units=max_class_id)

# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=current_y_batch))

# optimization
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


init = tf.global_variables_initializer()


# session
with tf.Session() as sess:
	sess.run(init)


	for i in range(n_epoch):
		print('Epoch: %d' % i)
		j=1
		sess.run(iterator.initializer)
		while True:
			try:
				results = sess.run((loss, 
									outputs, 
									last_step_output, 
									tf.stack([current_seq_len_batch-1, tf.range(tf.shape(current_X_batch)[0])], axis=1), 
									current_X_sequence_batch, 
									current_y_batch, 
									current_seq_len_batch,
									current_batch))
				print('Batch: %d' % j)
				pprint(results[7])
				j+=1
				#_, current_loss = sess.run((optimizer, loss))					
				#print('Loss: %f' % current_loss)
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


