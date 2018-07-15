import tensorflow as tf
import numpy as np
import pickle
from pprint import pprint
from sklearn.model_selection import train_test_split




#-----------------------------------------------------------------
#--------------------------FAKE DATASET---------------------------
#-----------------------------------------------------------------

# ========fake parameters=======
max_class_id = 5 # y_true = activity
max_n_frame = 400 # max_n_frames
max_freq = 3 # obj_freq_in_frame
n_feature = 33 # bag-of-objects
n_video = 22


# ============creation==========
videos_detection = []

for i in range(n_video):
	current_class_id = np.random.randint(0,max_class_id)
	current_n_frame = np.random.randint(1,max_n_frame+1)
	#current_n_frame = 40

	current_sequence=[]
	for j in range(current_n_frame):
		current_frame = np.random.randint(0,
										  high=max_freq,
										  size=(1,n_feature), 
										  dtype=np.uint8).tolist()[0]
		current_sequence.append(current_frame)

	videos_detection.append({'sequence':current_sequence, 
							 'activity_id':current_class_id,
							 'n_frame':current_n_frame})


pickle.dump(videos_detection, open('videos_detection.pickle', 'wb'))


videos_detection = pickle.load(open('videos_detection.pickle', 'rb'))


#*********************************
# TODO: DATA PREPROCESSING STEPS
#*********************************

#============preparation===========

X,y,seq_len=[],[],[]

for index,i in enumerate(videos_detection):
	X.append([frame_detection for frame_detection in i['sequence']])
	one_hot = [0]*max_class_id
	one_hot[i['activity_id']-1] = 1
	y.append(one_hot)
	seq_len.append(i['n_frame'])


#==========splitting==============
X_train, X_test, y_train, y_test, seq_len_train, seq_len_test = \
	 train_test_split(X,y,seq_len,test_size=0.3, random_state=0)#, stratify=y)


print(len(X_train))




#-----------------------------------------------------------------------------
#------------------------------------NETWORK----------------------------------
#-----------------------------------------------------------------------------

# NN params
lstm_in_cell_units=5 # design choice (hyperparameter)

# training params
n_epoch = 40
train_batch_size=3
train_fakebatch_size = len(X_train)
test_fakebatch_size = len(X_test)
learning_rate=0.01
# ********************************************************
#!!!!IMPORTANTEEEEE!!!
# handling last batch remainder
n_iteration = len(X_train)//train_batch_size
print(n_iteration)
# *********************************************************

zipped_train_data = list(zip(X_train,y_train,seq_len_train))
zipped_test_data = list(zip(X_test,y_test,seq_len_test))




#=========================graph===========================

lstmstate_batch_size = tf.placeholder(tf.int32, shape=[])

# dataset
train_data = tf.data.Dataset.from_generator(lambda: zipped_train_data, (tf.int32, tf.int32, tf.int32))
test_data = tf.data.Dataset.from_generator(lambda: zipped_test_data, (tf.int32, tf.int32, tf.int32))

# shuffle (whole) train_data
train_data = train_data.shuffle(buffer_size=len(X))

# obtain a padded_batch (recall that we are working with sequences!)
shape = ([None,n_feature],[max_class_id],[])
train_data_batch = train_data.padded_batch(train_batch_size, padded_shapes=shape)
# fake batches, they're the entire train and test dataset -> just needed to pad them!
# they will be used in the validation phase (not for training)
train_data_fakebatch = train_data.padded_batch(train_fakebatch_size, padded_shapes=shape) 
test_data_fakebatch = test_data.padded_batch(test_fakebatch_size, padded_shapes=shape) 

# iterator structure(s) - it is needed to make a reinitializable iterator (TF docs) -> dataset parametrization (without placeholders)
iterator = tf.data.Iterator.from_structure(train_data_batch.output_types, train_data_batch.output_shapes)


# this is the op that makes the magic -> dataset parametrization
train_iterator_init = iterator.make_initializer(train_data_batch)
faketrain_iterator_init = iterator.make_initializer(train_data_fakebatch)
faketest_iterator_init = iterator.make_initializer(test_data_fakebatch)


# so, to be even clearer, this is a "parameterized" op and its output depends on the particular iterator 
# initialization op executed before it during the session
# therefore from now on all the ops in the graph are "parameterized" -> not specialized on train or test
# IN OTHER WORDS, THE DATASET NOW BECOMES A PARAMETER THAT WE CAN SET DURING THE SESSION PHASE
# THANKS TO THE EXECUTION OF THE OP train_iterator_init OR test_iterator_init BEFORE THE EXECUTION OF THE OP next_batch
next_batch = iterator.get_next()

# split the batch in X, y, seq_len
# they will be singularily used in different ops
current_X_batch = tf.cast(next_batch[0], dtype=tf.float32)
current_y_batch = next_batch[1]
current_seq_len_batch = tf.reshape(next_batch[2], (1,-1))[0]

# lstm
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_in_cell_units, state_is_tuple=True)

#state_c, state_h = lstm_cell.zero_state(lstmstate_batch_size, tf.float32)
#initial_state = tf.nn.rnn_cell.LSTMStateTuple(tf.Variable(state_c, trainable=False), tf.Variable(state_h, trainable=False))

initial_state = lstm_cell.zero_state(lstmstate_batch_size, tf.float32)

outputs, states = tf.nn.dynamic_rnn(lstm_cell, current_X_batch, initial_state=initial_state, sequence_length=current_seq_len_batch, dtype=tf.float32)

# last_step_output done right (each instance will have it's own seq_len therefore the right last ouptut for each instance must be taken)
last_step_output = tf.gather_nd(outputs, tf.stack([tf.range(tf.shape(current_X_batch)[0]), current_seq_len_batch-1], axis=1))

# logits
logits = tf.layers.dense(last_step_output, units=max_class_id)

# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=current_y_batch))

# optimization (only during training phase (OBVIOUSLY))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(current_y_batch, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

init = tf.global_variables_initializer()




#==========================session==========================

with tf.Session() as sess:
	sess.run(init)
	
	#***************** TRAINING ************************
	for i in range(n_epoch):
		print('\nEpoch: %d/%d' % ((i+1), n_epoch))
		sess.run(train_iterator_init)
		
		for j in range(n_iteration):
			_, batch_loss = sess.run((optimizer, loss), feed_dict={lstmstate_batch_size:train_batch_size})					
			print('Batch: %d/%d - Loss: %f' % ((j+1), n_iteration, batch_loss))

		
			# results = sess.run((current_X_batch, 
			# 					current_y_batch, 
			# 					current_seq_len_batch))
			# pprint(results[0].shape)


		#****************** VALIDATION ******************
		# end of every epoch
		sess.run(faketrain_iterator_init)
		train_loss, train_acc = sess.run((loss, accuracy),feed_dict={lstmstate_batch_size:train_fakebatch_size})
		print('\nTrain_loss: %f' % train_loss)
		print('Train_acc: %f' % train_acc)

		sess.run(faketest_iterator_init)
		test_loss, test_acc = sess.run((loss, accuracy),feed_dict={lstmstate_batch_size:test_fakebatch_size})
		print('Test_loss: %f' % train_loss)
		print('Test_acc: %f' % train_acc)






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
