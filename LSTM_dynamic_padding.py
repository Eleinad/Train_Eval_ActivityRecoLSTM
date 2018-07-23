import tensorflow as tf
import numpy as np
import pickle
from pprint import pprint
from sklearn.model_selection import train_test_split
import os
import time
import datetime



#-----------------------------------------------------------------
#--------------------------FAKE DATASET---------------------------
#-----------------------------------------------------------------
'''
# ========fake parameters=======
max_class_id = 5 # y_true = activity
max_n_frame = 7 # max_n_frames
max_freq = 3 # obj_freq_in_frame
n_feature = 5 # bag-of-objects
n_video = 22

# ============creation==========

dataset_detection_video = []

for i in range(n_video):
    current_class_id = np.random.randint(0,max_class_id)
    current_n_frame = np.random.randint(10,10+max_n_frame)
    #current_n_frame = 400
    
    current_objs_in_frame = []
    for j in range(current_n_frame):
        current_n_objs_in_frame = np.random.randint(1,5)
        current_frame = np.random.choice(n_feature, current_n_objs_in_frame, replace=True)
        current_frame += 1
        current_objs_in_frame.append({'obj_class_ids': current_frame,
        							  'obj_rois':0})

    dataset_detection_video.append({'frames_info':current_objs_in_frame,
                                    'class_id':current_class_id,
                                    'reduced_fps':4,
                                    'final_nframes':len(current_objs_in_frame)})


pickle.dump(dataset_detection_video, open('dataset_detection_video.pickle', 'wb'))

dataset_detection_video = pickle.load(open('dataset_detection_video.pickle', 'rb'))
'''

max_class_id = 7 # y_true = activity
n_feature = 33 # bag-of-objects

pickle_path = './PersonalCare/pickle'
dataset_detection_video = [pickle.load(open(pickle_path+'/'+video_pickle,'rb')) for video_pickle in os.listdir(pickle_path)]
    
classlbl_to_classid = {} 
classid = 0

for i in dataset_detection_video:
    classlbl = i['class_id'].lower().replace(' ','')
    if classlbl not in classlbl_to_classid:
        classlbl_to_classid[classlbl] = classid
        classid += 1

    i['class_id'] = classlbl_to_classid[classlbl]


classid_to_classlbl = {value:key for key,value in classlbl_to_classid.items()}



# videos must be at least 5 s long
dataset_detection_video = [i for i in dataset_detection_video if (i['final_nframes']//i['reduced_fps']) >= 5 and classid_to_classlbl[i['class_id']] != 'washingface']



#==================FEATURE BAG-OF-OBJS===============

dataset_boo_video = []

for video in dataset_detection_video:
    
    video_boo_matrix = np.zeros((video['final_nframes'],n_feature), dtype=np.uint8)

    for index, frame in enumerate(video['frames_info']) :
    	boo = {}

    	for obj in frame['obj_class_ids']:
    		if obj not in boo:
    			boo[obj] = 1
    		else:
    			boo[obj] += 1

    	for class_id_index, obj_freq in boo.items():
    		video_boo_matrix[index][class_id_index-1] = obj_freq

      
    dataset_boo_video.append({'class_id': video['class_id'],
                              'final_nframes': video['final_nframes'],
                              'reduced_fps':video['reduced_fps'],
                              'sequence': video_boo_matrix})



#==================CO-OCC FREQ OBJS================


dataset_cooc_video = []

for video in dataset_boo_video:
	n_frame = video['final_nframes']
	n_batch = 5*video['reduced_fps']

	iteration = int(n_frame//(n_batch//2))
	cooc_flat_seq_matrix = np.zeros((iteration, (n_feature)*(n_feature+1)//2), dtype=np.uint8)


	for i in range(iteration):
		if n_batch+((n_batch//2)*i) <= n_frame:
			end = int(n_batch+((n_batch//2)*i))
		else:
			end = n_frame

		frame_batch = video['sequence'][int(n_batch//2)*i:end,:]
		frame_batch = np.where(frame_batch>0,1,0)
		cooc_tri_upper = np.triu(frame_batch.T @ frame_batch)

		cooc_flat_index = 0
		for j in range(n_feature):
			for k in range(j,n_feature):
				cooc_flat_seq_matrix[i, cooc_flat_index] = cooc_tri_upper[j,k]
				cooc_flat_index+=1

	video['sequence'] = cooc_flat_seq_matrix
	dataset_cooc_video.append(video)




#============final transformation (sequence and one_hot)===========


X,y,seq_len=[],[],[]

for index,i in enumerate(dataset_boo_video):
	X.append([frame_detection.tolist() for frame_detection in i['sequence']])
	one_hot = [0]*max_class_id
	one_hot[i['class_id']-1] = 1
	y.append(one_hot)
	seq_len.append(i['sequence'].shape[0])



#==========splitting==============
X_train, X_test, y_train, y_test, seq_len_train, seq_len_test = \
	 train_test_split(X,y,seq_len,test_size=0.3, random_state=0, stratify=y)


print(len(X_train))
print(len(X_test))




#-----------------------------------------------------------------------------
#------------------------------------NETWORK----------------------------------
#-----------------------------------------------------------------------------

# NN params
lstm_in_cell_units=20 # design choice (hyperparameter)

# training params
n_epoch = 100
n_layer = 2
train_batch_size=32
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
shape = ([None,len(X[0][0])],[max_class_id],[])
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
lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*n_layer, state_is_tuple=True)
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



losses = {
		  'train_loss':[],
		  'train_acc':[],
		  'test_loss':[],
		  'test_acc':[]
		  }


#==========================session==========================

with tf.Session() as sess:
	sess.run(init)
	
	#***************** TRAINING ************************

	for i in range(n_epoch):
		start_epoch_time = time.time()
		print('\nEpoch: %d/%d' % ((i+1), n_epoch))
		sess.run(train_iterator_init)
		
		for j in range(n_iteration):
			start_batch_time = time.time()
			_, batch_loss = sess.run((optimizer, loss), feed_dict={lstmstate_batch_size:train_batch_size})
			batch_time = str(datetime.timedelta(seconds=round(time.time()-start_batch_time, 2)))
			print('Batch: %d/%d - Loss: %f - Time: %s' % ((j+1), n_iteration, batch_loss, batch_time))

		
			# results = sess.run((current_X_batch, 
			# 					current_y_batch, 
			# 					current_seq_len_batch))
			# pprint(results[0].shape)


		#****************** VALIDATION ******************
		epoch_time = str(datetime.timedelta(seconds=round(time.time()-start_epoch_time, 2)))
		print('Tot epoch time: %s' % (epoch_time))
		# end of every epoch
		sess.run(faketrain_iterator_init)
		train_loss, train_acc = sess.run((loss, accuracy),feed_dict={lstmstate_batch_size:train_fakebatch_size})
		print('\nTrain_loss: %f' % train_loss)
		print('Train_acc: %f' % train_acc)


		sess.run(faketest_iterator_init)
		test_loss, test_acc = sess.run((loss, accuracy),feed_dict={lstmstate_batch_size:test_fakebatch_size})
		print('Test_loss: %f' % test_loss)
		print('Test_acc: %f' % test_acc)

		losses['train_loss'].append(train_loss)
		losses['train_acc'].append(train_acc)
		losses['test_loss'].append(test_loss)
		losses['test_acc'].append(test_acc)

	pickle.dump(losses, open('losses.pickle','wb'))





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
