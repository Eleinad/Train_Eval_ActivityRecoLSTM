import tensorflow as tf
import time
import datetime
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import numpy as np
import os



learning_rate=0.0005



def graph(splitted_data, lstm_in_cell_units=5, relu_units=5):

	X_train = splitted_data[0]
	X_test = splitted_data[1]
	y_train = splitted_data[2]
	y_test = splitted_data[3]
	seq_len_train =splitted_data[4]
	seq_len_test = splitted_data[5]


	lstm_in_cell_units=lstm_in_cell_units # design choice (hyperparameter)
	relu_units=relu_units
	train_fakebatch_size = len(X_train)
	test_fakebatch_size = len(X_test)

	zipped_train_data = list(zip(X_train,y_train,seq_len_train))
	zipped_test_data = list(zip(X_test,y_test,seq_len_test))


	#=========================graph===========================
	#tf.set_random_seed(1234)
	tf.reset_default_graph()

	batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')

	# dataset
	train_data = tf.data.Dataset.from_generator(lambda: zipped_train_data, (tf.int32, tf.int32, tf.int32))
	test_data = tf.data.Dataset.from_generator(lambda: zipped_test_data, (tf.int32, tf.int32, tf.int32))

	# shuffle (whole) train_data
	train_data = train_data.shuffle(buffer_size=len(X_train))

	# obtain a padded_batch (recall that we are working with sequences!)
	shape = ([None,len(X_train[0][0])],[len(y_train[0])],[])
	train_data_batch = train_data.padded_batch(tf.cast(batch_size,tf.int64), padded_shapes=shape)
	# fake batches, they're the entire train and test dataset -> just needed to pad them!
	# they will be used in the validation phase (not for training)
	train_data_fakebatch = train_data.padded_batch(tf.cast(batch_size,tf.int64), padded_shapes=shape) 
	test_data_fakebatch = test_data.padded_batch(tf.cast(batch_size,tf.int64), padded_shapes=shape) 

	# iterator structure(s) - it is needed to make a reinitializable iterator (TF docs) -> dataset parametrization (without placeholders)
	iterator = tf.data.Iterator.from_structure(train_data_batch.output_types, train_data_batch.output_shapes)


	# this is the op that makes the magic -> dataset parametrization
	train_iterator_init = iterator.make_initializer(train_data_batch, name='train_iterator_init')
	faketrain_iterator_init = iterator.make_initializer(train_data_fakebatch, name='faketrain_iterator_init')
	faketest_iterator_init = iterator.make_initializer(test_data_fakebatch, name='faketest_iterator_init')


	# so, to clarify, this is a "parameterized" op and its output depends on the particular iterator 
	# initialization op executed before it during the session
	# therefore from now on all the ops in the graph are "parameterized" -> not specialized on train or test
	# IN OTHER WORDS, THE DATASET NOW BECOMES A PARAMETER THAT WE CAN SET DURING THE SESSION PHASE
	# THANKS TO THE EXECUTION OF THE OP train_iterator_init OR test_iterator_init BEFORE THE EXECUTION OF THE OP next_batch
	next_batch = iterator.get_next()

	# split the batch in X, y, seq_len
	# they will be singularly used in different ops
	current_X_batch = tf.cast(next_batch[0], dtype=tf.float32)
	current_y_batch = next_batch[1]
	current_seq_len_batch = tf.reshape(next_batch[2], (1,-1))[0]

	# lstm
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_in_cell_units, state_is_tuple=True)
	#state_c, state_h = lstm_cell.zero_state(batch_size, tf.float32)
	#initial_state = tf.nn.rnn_cell.LSTMStateTuple(tf.Variable(state_c, trainable=False), tf.Variable(state_h, trainable=False))
	initial_state = lstm_cell.zero_state(batch_size, tf.float32)
	_, states = tf.nn.dynamic_rnn(lstm_cell, current_X_batch, initial_state=initial_state, sequence_length=current_seq_len_batch, dtype=tf.float32)

	#relu = tf.layers.dense(inputs=states[1], units=relu_units, activation=tf.nn.relu, name='relu')

	# last_step_output done right (each instance will have it's own seq_len therefore the right last ouptut for each instance must be taken)
	#last_step_output = tf.gather_nd(outputs, tf.stack([tf.range(tf.shape(current_X_batch)[0]), current_seq_len_batch-1], axis=1))

	# logits
	#hidden_state = output per cui last_step_output è superfluo, grazie a current_seq_len_batch ritorna l'hidden_state del giusto timestep
	#states è una tupla (cell_state, hidden_state) dell'ultimo timestep (in base a current_seq_len_batch)
	logits = tf.layers.dense(states[1], units=len(y_train[0]), name='logits')

	# loss
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=current_y_batch), name='loss')

	# optimization (only during training phase (OBVIOUSLY))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, name='optimizer')

	# ops for accuracy and confusion matrix 
	y_pred = tf.argmax(logits, 1)
	y_true = tf.argmax(current_y_batch, 1)
	correct_pred = tf.equal(y_pred, y_true)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32), name='accuracy')

	init = tf.global_variables_initializer()

	# debugging & training visualization
	all_variables = tf.global_variables()

	for i in all_variables:
		tf.summary.histogram(i.name.replace(':','_'), i)

	summaries = tf.summary.merge_all()

	# building collections
	train_op_list = [init,
					 batch_size,
					 train_iterator_init,
					 faketrain_iterator_init,
					 faketest_iterator_init,
					 logits,
					 loss,
					 optimizer,
					 summaries,
					 y_true,
					 y_pred,
					 accuracy]
	
	for op in train_op_list:
		tf.get_default_graph().add_to_collection('my_train_op',op)

	inference_op_list = [batch_size,
						 y_true,
						 y_pred]
	
	for op in inference_op_list:
		tf.get_default_graph().add_to_collection('my_inference_op',op)

	hyperparameters = [lstm_in_cell_units, relu_units]
	for hyp in hyperparameters:
		tf.get_default_graph().add_to_collection('hyperparameters', hyp)

	# exporting graph
	tf.train.export_meta_graph(filename='./graph/graph.meta')



def train(splitted_data, classlbl_to_classid, n_epoch, train_batch_size, feat_type):

	graph=tf.get_default_graph()

	#fetching graph training op from the collection
	train_op_list = graph.get_collection('my_train_op')
	init = train_op_list[0]
	batch_size = train_op_list[1]
	train_iterator_init = train_op_list[2]
	faketrain_iterator_init = train_op_list[3]
	faketest_iterator_init = train_op_list[4]
	logits = train_op_list[5]
	loss = train_op_list[6]
	optimizer = train_op_list[7]
	summaries = train_op_list[8]
	y_true = train_op_list[9]
	y_pred = train_op_list[10]
	accuracy = train_op_list[11]
	saver = tf.train.Saver(max_to_keep=3)


	#fetching graph hyperparameters from the collection
	hyperparameters = graph.get_collection('hyperparameters')
	lstm_in_cell_units = hyperparameters[0]
	relu_units = hyperparameters[1]

	losses = {
		  'train_loss':[],
		  'train_acc':[],
		  'test_loss':[],
		  'test_acc':[],
		  'top_3_acc':[],
		  'top_5_acc':[]
		  }

	X_train = splitted_data[0]
	X_test = splitted_data[1]

	faketrain_batch_size = len(X_train)
	faketest_batch_size = len(X_test)

	n_iteration = len(X_train)//train_batch_size



	with tf.Session() as sess:
		
		sess.run(init)	
		writer = tf.summary.FileWriter("events_W_histograms")
		

		#***************** TRAINING ********************
		for i in range(n_epoch):
			writer.add_summary(sess.run(summaries), global_step=i)
			
			start_epoch_time = time.time()
			print('\nEpoch: %d/%d' % ((i+1), n_epoch))
			sess.run(train_iterator_init, feed_dict={batch_size:train_batch_size})
			
			for j in range(n_iteration):

				start_batch_time = time.time()
				_, batch_loss = sess.run((optimizer, loss), feed_dict={batch_size:train_batch_size})
				batch_time = str(datetime.timedelta(seconds=round(time.time()-start_batch_time, 2)))
				print('Batch: %d/%d - Loss: %f - Time: %s' % ((j+1), n_iteration, batch_loss, batch_time))

				# print('Batch')
				# results = sess.run((#current_X_batch, 
				# 					#current_y_batch, 
				# 					#current_seq_len_batch,
				# 					states), feed_dict={lstmstate_batch_size:train_batch_size})
				# print(results[0][1])


			epoch_time = str(datetime.timedelta(seconds=round(time.time()-start_epoch_time, 2)))
			print('Tot epoch time: %s' % (epoch_time))

			save_path = saver.save(sess, "./weights/model.ckpt", global_step=i, write_meta_graph=False)


			#****************** VALIDATION (after each epoch) ******************
			# whole training set
			sess.run(faketrain_iterator_init, 
					 feed_dict={batch_size:faketrain_batch_size})
			train_loss, train_acc = sess.run((loss, accuracy),
											 feed_dict={batch_size:faketrain_batch_size})
			print('\nTrain_loss: %f' % train_loss)
			print('Train_acc: %f' % train_acc)

			# whole test set
			sess.run(faketest_iterator_init, 
					 feed_dict={batch_size:faketest_batch_size})
			test_loss, test_acc, logits_, y_true_ = sess.run((loss, accuracy, logits, y_true),
											feed_dict={batch_size:faketest_batch_size})
			print('Test_loss: %f' % test_loss)
			print('Test_acc: %f' % test_acc)

			# TOP-3 and TOP-5
			ordered_logits = []

			for i in logits_:
				temp = []
				for index,j in enumerate(i):
					temp.append((index,j))
				temp.sort(key=lambda x:x[1], reverse=True)
				ordered_logits.append(temp)


			top_3 = [1 if k in [j[0] for j in i[:3]] else 0 for i,k in zip(ordered_logits,y_true_)]
			top_3_acc = sum(top_3)/len(top_3)
			top_5 = [1 if k in [j[0] for j in i[:5]] else 0 for i,k in zip(ordered_logits,y_true_)]
			top_5_acc = sum(top_5)/len(top_5)

			print('Top_3_acc: %f' % top_3_acc)
			print('Top_5_acc: %f' % top_5_acc)

			losses['train_loss'].append(train_loss)
			losses['train_acc'].append(train_acc)
			losses['test_loss'].append(test_loss)
			losses['test_acc'].append(test_acc)
			losses['top_3_acc'].append(top_3_acc)
			losses['top_5_acc'].append(top_5_acc)


		#************* CONFUSION MATRIX *************
		sess.run(faketest_iterator_init, 
				 feed_dict={batch_size:faketest_batch_size})
		test_y_true, test_y_pred = sess.run((y_true, y_pred),feed_dict={batch_size:faketest_batch_size})

		for i in tf.get_default_graph().get_collection('trainable_variables'):
			print(sess.run(i)[0])


		print()
		print(classlbl_to_classid)
		print()
		print(confusion_matrix(test_y_true, test_y_pred, labels=[0,1,2,3,4,5,6]))
		print()
		print(classification_report(test_y_true, test_y_pred, labels=[0,1,2,3,4,5,6]))
		#print()
		#misclassified_nframe = [seq_len[i[0]]*n_batch for i in np.argwhere(np.equal(test_y_true,test_y_pred)==False)]
		#print(misclassified_nframe)

		loss_dir = './loss'
		if not os.path.exists(loss_dir):
			os.makedirs(loss_dir)
		pickle.dump(losses, open(loss_dir+'/'+'losses_'+feat_type+'_'+str(lstm_in_cell_units)+'_'+str(relu_units)+'.pickle','wb'))







def predict(X, y, seq):

	fakeinference_batch_size = len(X)
	zipped_inference_data = list(zip(X,y,seq))

	tf.reset_default_graph()
	tf.train.import_meta_graph('./tmp/graph.meta')
	#print(tf.get_default_graph().collections)
	#print(tf.get_default_graph().get_collection('trainable_variables'))
	pretrained_weights = tf.get_default_graph().get_collection('trainable_variables')
	print(pretrained_weights)
	tf.reset_default_graph()


	inference_data = tf.data.Dataset.from_generator(lambda: zipped_inference_data, (tf.int32, tf.int32, tf.int32))

	shape = ([None,len(X[0][0])],[len(y[0])],[])
	inference_data_fakebatch = inference_data.padded_batch(fakeinference_batch_size, padded_shapes=shape) 

	#iterator = tf.data.Iterator.from_structure(inference_data_fakebatch.output_types, inference_data_fakebatch.output_shapes)

	#fakeinference_iterator_init = iterator.make_initializer(inference_data_fakebatch, name='fakeinference_iterator_init')

	iterator = inference_data_fakebatch.make_one_shot_iterator()

	next_batch = iterator.get_next()

	current_X_batch = tf.cast(next_batch[0], dtype=tf.float32)
	current_y_batch = next_batch[1]
	current_seq_len_batch = tf.reshape(next_batch[2], (1,-1))[0]

	# lstm
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(20, state_is_tuple=True)
	initial_state = lstm_cell.zero_state(fakeinference_batch_size, tf.float32)
	_, states = tf.nn.dynamic_rnn(lstm_cell, current_X_batch, initial_state=initial_state, sequence_length=current_seq_len_batch, dtype=tf.float32)

	# logits
	logits = tf.layers.dense(states[1], units=7, name='logits')

	# ops for accuracy and confusion matrix 
	y_pred = tf.argmax(logits, 1)
	y_true = tf.argmax(current_y_batch, 1)
	correct_pred = tf.equal(y_pred, y_true)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32), name='accuracy')

	graph=tf.get_default_graph()
	#print(tf.get_default_graph().collections)
	#print(tf.get_default_graph().get_collection('trainable_variables'))
	

	# restoring pretrained weights
	# restore_w_ops = []
	# for i,j in zip(graph.get_collection('trainable_variables'),pretrained_weights):
	# 	scope = i.name[0:i.name.rfind('/')]
	# 	print(scope, i.name[i.name.rfind('/')+1:i.name.find(':')])
	# 	with tf.variable_scope(scope, reuse=True):
	# 		restore_w_ops.append(tf.get_variable(i.name[i.name.rfind('/')+1:i.name.find(':')],  
	# 							 initializer=j))


	with tf.variable_scope('rnn/basic_lstm_cell', reuse=True):
		a = tf.get_variable('kernel',  initializer=np.ones((53,80)))

	# with tf.variable_scope('prova'):
	# 	b = tf.get_variable('kernel',  initializer=pretrained_weights[0].initialized_value())

	# with tf.variable_scope('prova'):
	# 	a = tf.Variable(pretrained_weights[0], name='kernel')



	with tf.Session() as sess:
		# a = sess.run(tf.report_uninitialized_variables())
		# print(a)
		# # saver.restore(sess, tf.train.latest_checkpoint('./tmp'))
		# for i,j in zip(graph.get_collection('trainable_variables'),pretrained_trainable_variables):
		# 	value = sess.run(i)
		# 	i.load(value, sess)
		#sess.run(init)
		from pprint import pprint
		pprint(tf.get_default_graph().get_collection('variables'))
		sess.run(a.initializer)
		print(sess.run(tf.report_uninitialized_variables()))
		for i in tf.get_default_graph().get_collection('variables'):
			if 'kernel' in i.name and ('rnn' in i.name or 'prova' in i.name):
				print(i.name)
				print(sess.run(i)[0])
		#for op in restore_w_ops:
		
		b = sess.run(tf.report_uninitialized_variables())
		print(b)
		#test_y_true, test_y_pred, accuracy = sess.run((y_true, y_pred, accuracy))

	#print(test_y_true, test_y_pred, accuracy) 