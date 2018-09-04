import tensorflow as tf
import numpy as np
import pickle
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
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





#-----------------------------------------------------------------
#--------------------------TRUE DATASET---------------------------
#-----------------------------------------------------------------


#=============loading data==============

pickle_path = './PersonalCare/pickle'
dataset_detection_video = [pickle.load(open(pickle_path+'/'+video_pickle,'rb')) for video_pickle in os.listdir(pickle_path) if 'face' not in pickle.load(open(pickle_path+'/'+video_pickle,'rb'))['class_id']]
    
classlbl_to_classid = {} 
classid = 0

for i in dataset_detection_video:
    classlbl = i['class_id'].lower().replace(' ','')
    if classlbl not in classlbl_to_classid:
        classlbl_to_classid[classlbl] = classid
        classid += 1

    i['class_id'] = classlbl_to_classid[classlbl]


classid_to_classlbl = {value:key for key,value in classlbl_to_classid.items()}


print(classid_to_classlbl)

# filtering data -> videos must be at least 5 s
dataset_detection_video = [i for i in dataset_detection_video if (i['final_nframes']//i['reduced_fps']) >= 5]


print('Full dataset len %d' % len(dataset_detection_video))


#============true parameters==========

max_class_id = 7 # y_true = activity
n_feature = 33 # bag-of-objects

'''

#==================BAG-OF-TF-IDFSUBSETOBJS===============

dataset_boo_video = []

mapping = {5:0,6:1,7:2,8:3,10:4,12:5,33:6}

for video in dataset_detection_video:
    
    video_boo_matrix = np.zeros((video['final_nframes'],n_feature), dtype=np.uint8)

    for index, frame in enumerate(video['frames_info']) :
    	boo = {}

    	for obj in frame['obj_class_ids']:
    		if obj in list(mapping.keys()):
	    		if obj not in boo:
	    			boo[obj] = 1
	    		else:
	    			boo[obj] += 1

    	for class_id_index, obj_freq in boo.items():
    		video_boo_matrix[index][mapping[class_id_index]] = obj_freq

    video_boo_matrix = video_boo_matrix[~np.all(video_boo_matrix == 0, axis=1)]

    dataset_boo_video.append({'class_id': video['class_id'],
                              'final_nframes': video['final_nframes'],
                              'reduced_fps':video['reduced_fps'],
                              'sequence': video_boo_matrix})

    # filtro i video che hanno una sequence length minore del batch length
    dataset_boo_video = [i for i in dataset_boo_video if i['sequence'].shape[0]>=9]

'''


#==================BAG-OF-OBJS===============

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






#==============BATCHED BAG-OF-OBJS============

dataset_batchedboo_video = []



for video in dataset_boo_video:

	n_frame = video['final_nframes']
	n_batch = 9

	video_batchedboo_matrix = np.zeros((int(n_frame/n_batch),n_feature))

	iteration = int(n_frame/n_batch)

	for i in range(iteration):
		frame_batch = video['sequence'][(n_batch*i):((n_batch*i)+n_batch),:]
		video_batchedboo_matrix[i] = np.sum(frame_batch, axis=0)

	dataset_batchedboo_video.append({'class_id': video['class_id'],
                              'final_nframes': video['final_nframes'],
                              'reduced_fps':video['reduced_fps'],
                              'sequence': video_batchedboo_matrix})	



# l = []
# for video_b in dataset_batchedboo_video:

# 	n_b = video_b['sequence'].shape[0]*video_b['sequence'].shape[1]
# 	l.append([(n_b-np.count_nonzero(video_b['sequence']))*100/n_b])



from sklearn.cluster import KMeans


sequences = dataset_batchedboo_video[0]['sequence']
for i in range(1,len(dataset_batchedboo_video)):
	sequences = np.vstack((sequences,dataset_batchedboo_video[i]['sequence']))

print(sequences.shape)

kmeans = KMeans(n_clusters=100, random_state=0, n_jobs=-1).fit(sequences)
labels = list(kmeans.labels_)
codebook = list(kmeans.cluster_centers_)

for video in dataset_batchedboo_video:
	curr_seq_len = video['sequence'].shape[0]
	curr_labels = labels[:curr_seq_len]
	for j in range(curr_seq_len):
		video['sequence'][j,:] = codebook[curr_labels[j]]
	labels = labels[curr_seq_len:]




'''
#================AVG-SPEED and AVG-VELOCITY=========================

def inside(start, end, c_start, c_end):
	frame_batch_range = set(range(start,end+1))
	contiguous_range = set(range(c_start, c_end+1))

	if len(frame_batch_range.intersection(contiguous_range)) > 0:
		return 1
	else:
		return 0

def centroid_roi(roi):
	return (roi[2]+roi[0])/2, (roi[3]+roi[1])/2


dataset_batchedvelocity_video, dataset_batchedspeed_video, prova = [], [], []

for video in dataset_detection_video:

	# costruzione della struttura dati contenente i centroidi degli oggetti nei frame
	centroids_list = []
	for frame in video['frames_info']:
		centroids_list.append([[] for _ in range(33)])
		objs = frame['obj_class_ids']
		rois = frame['obj_rois']
		for i in range(objs.shape[0]):
			curr_obj_roi = rois[i]
			curr_obj_id = objs[i]-1
			(x, y) = centroid_roi(curr_obj_roi)
			centroids_list[-1][curr_obj_id].append((int(x),int(y)))



	# encoding di centroids_list in una binary matrix
	# da usare dopo per ottenere objid_to_contiguous_intervals
	n = video['final_nframes']

	all_objs = set({})
	for i in range(n):
		objs = video['frames_info'][i]['obj_class_ids']
		all_objs = all_objs.union(set(objs))

	all_objs = sorted(list(all_objs))

	binary_sequence = np.zeros((len(centroids_list),33), dtype=np.uint8)

	for i in all_objs:
		for index,j in enumerate(centroids_list):
			if len(j[i-1]) != 0: #basta che sia presente almeno una volta
				binary_sequence[index,i-1] = 1


	#img = Image.fromarray(binary_sequence.astype(np.uint8)*255)
	#img.show()


	# costruzione di objid_to_contiguous_intervals
	binary_sequence = np.vstack([binary_sequence,np.repeat(2,33)])
	objid_to_contiguous_intervals = {}

	for i in all_objs:
	    contiguous_intervals = []
	    t_zero, t_uno = 2, 2
	    for index,curr_value in enumerate(binary_sequence[:,i-1]):
	        t_due = t_uno
	        t_uno = t_zero
	        t_zero = curr_value
	        if (t_due,t_uno,t_zero)==(0,1,1) or (t_due,t_uno,t_zero)==(2,1,1):
	            temp=[]
	            temp.append(index-1)
	        elif (t_due,t_uno,t_zero)==(1,1,0) or (t_due,t_uno,t_zero)==(1,1,2):
	            temp.append(index-1)
	            temp.append(temp[1]-temp[0]+1)
	            contiguous_intervals.append(list(temp))

	    objid_to_contiguous_intervals[i] = contiguous_intervals



	# costruzione di objid_to_listavgspeedincontiguous
	# calcolo della avg speed per ogni continguo sfruttando objid_to_contiguous_intervals
	objid_to_listavgspeedincontiguous = {}

	for i in objid_to_contiguous_intervals.keys():
		if len(objid_to_contiguous_intervals[i])>0:
			objid_to_listavgspeedincontiguous[i] = []
			curr_obj_contiguous_list = objid_to_contiguous_intervals[i]
			for j in curr_obj_contiguous_list:
				coord_list = []
				start_frame = j[0]
				end_frame = j[1]
				frame_length = j[2]
				start_coord = (centroids_list[j[0]][i-1][0], 0) #se ce n'è più di uno seleziona il primo
				coord_list.append(start_coord)
				for k in range(start_frame+1,end_frame+1):
					temp = []
					for index,next_centroid in enumerate(centroids_list[k][i-1]): #se ce n'è più di uno seleziona quello più vicino
						euc_dist = np.sqrt(np.power(next_centroid[0]-coord_list[-1][0][0], 2) + np.power(next_centroid[1]-coord_list[-1][0][1], 2))
						#print(euc_dist)
						temp.append((index, euc_dist))
					temp.sort(key=lambda x: x[1])
					coord_list.append((centroids_list[k][i-1][temp[0][0]], coord_list[-1][1]+temp[0][1]))
					#print(coord_list)
				objid_to_listavgspeedincontiguous[i].append((coord_list[0][0], coord_list[-1][0], coord_list[-1][1]/frame_length, frame_length))




	# a questo punto abbiamo 2 strutture dati:
	# 1. objid_to_contiguous_intervals (dict)
	#	 .keys	= objid (int)
	#    .value = start, end, length degli intervalli contigui (list of lists)
	# 2. objid_to_listavgspeedincontiguous (dict)
	#    .keys	= objid (int)
	#	 .value = speed nel corrispettivo contiguo
	# sfruttando queste due vengono costruite le speed features


	n_frame = video['final_nframes']
	n_batch = 9

	video_batchedspeed_matrix = np.zeros((int(n_frame/n_batch),n_feature))

	video_batchedvelocity_matrix = np.zeros((int(n_frame/n_batch),n_feature*2))

	iteration = int(n_frame/n_batch)


	for i in range(iteration):
		temp = {}
		start_frame_batch = n_batch*i
		end_frame_batch = (n_batch*i)+n_batch

		for objid, contiguous_list in objid_to_contiguous_intervals.items():
			for c_index, contiguous in enumerate(contiguous_list):
				if inside(start_frame_batch, end_frame_batch, contiguous[0], contiguous[1]):
					temp[objid] = (np.subtract(objid_to_listavgspeedincontiguous[objid][c_index][1],objid_to_listavgspeedincontiguous[objid][c_index][0])/objid_to_listavgspeedincontiguous[objid][c_index][3], objid_to_listavgspeedincontiguous[objid][c_index][2]) # sostituisci sempre con l'ultimo
					#prova.append([i, objid, start_frame_batch, end_frame_batch, contiguous[0], contiguous[1], objid_to_listavgspeedincontiguous[objid][c_index]])

		for objid, values in temp.items():
			video_batchedspeed_matrix[i][objid-1] = values[1]
			video_batchedvelocity_matrix[i][objid-1] = values[0][0]
			video_batchedvelocity_matrix[i][objid] = values[0][1]

	
	dataset_batchedspeed_video.append({'class_id': video['class_id'],
                              'final_nframes': video['final_nframes'],
                              'reduced_fps':video['reduced_fps'],
                              'sequence': video_batchedspeed_matrix})
	dataset_batchedvelocity_video.append({'class_id': video['class_id'],
                              'final_nframes': video['final_nframes'],
                              'reduced_fps':video['reduced_fps'],
                              'sequence': video_batchedvelocity_matrix})



# minimum_speed = 0.0
# maximum_speed = 100.0

# for video in dataset_batchedspeed_video:
# 	video['sequence'] = np.where(video['sequence']>maximum_speed,maximum_speed,video['sequence'])


s = b = np.zeros((1,33))
max_s =  np.zeros((33,))
max_b = np.zeros((33,))
l = []
for video_s, video_b in zip(dataset_batchedspeed_video, dataset_batchedboo_video):
	n_s = video_s['sequence'].shape[0]*video_s['sequence'].shape[1]
	n_b = video_b['sequence'].shape[0]*video_b['sequence'].shape[1]
	l.append([(n_s-np.count_nonzero(video_s['sequence']))*100/n_s, (n_b-np.count_nonzero(video_b['sequence']))*100/n_b])
	s=s+np.count_nonzero(video_s['sequence'], axis=0)
	b=b+np.count_nonzero(video_b['sequence'], axis=0)

	for index,i in enumerate(np.max(video_s['sequence'], axis=0).astype(int)):
		if i>=max_s[index]:
			max_s[index] = i

	for index,i in enumerate(np.max(video_b['sequence'], axis=0).astype(int)):
		if i>=max_b[index]:
			max_b[index] = i

'''

'''
#================BATCHED BOO & NORM-SPEED MULTIPL======================




# # speed normalizing and frequency weighting
# for video_s, video_b in zip(dataset_batchedspeed_video, dataset_batchedboo_video):
# 	#video_s['sequence'] = video_s['sequence']/maximum_speed
# 	video_s['sequence'] = np.concatenate((video_s['sequence'],video_b['sequence']), axis=1)



'''

'''
#==================CO-OCC FREQ OBJS================

dataset_cooc_video = []

for video in dataset_boo_video:
	n_frame = video['final_nframes']
	n_batch = 3*video['reduced_fps']

	iteration = int(n_frame//(n_batch//2))
	cooc_flat_seq_matrix = np.zeros((iteration, (n_feature-1)*(n_feature+1-1)//2), dtype=np.uint8)


	for i in range(iteration):
		if n_batch+((n_batch//2)*i) <= n_frame:
			end = int(n_batch+((n_batch//2)*i))
		else:
			end = n_frame

		frame_batch = video['sequence'][int(n_batch//2)*i:end,:]
		frame_batch = np.where(frame_batch>0,1,0)
		cooc_tri_upper = np.triu(frame_batch.T @ frame_batch, 1)

		cooc_flat_index = 0
		for j in range(n_feature-1):
			for k in range((j+1),n_feature):
				cooc_flat_seq_matrix[i, cooc_flat_index] = cooc_tri_upper[j,k]
				cooc_flat_index+=1

	dataset_cooc_video.append({'class_id': video['class_id'],
                              'final_nframes': video['final_nframes'],
                              'reduced_fps':video['reduced_fps'],
                              'sequence': cooc_flat_seq_matrix})#np.where(cooc_flat_seq_matrix>0,1,0)



from sklearn.metrics.pairwise import cosine_similarity
results,mean,percent = [],[],[]
for video in dataset_cooc_video:
	results.append([cosine_similarity(video['sequence'][i+1].reshape(1,-1),video['sequence'][i].reshape(1,-1))[0][0] for i in range(video['sequence'].shape[0]-1)])
	mean.append(sum(results[-1])/len(results[-1]))
	nonzero = 0.0
	for i in video['sequence']:
		if np.count_nonzero(i) == 0:
			nonzero += 1.0
	percent.append(nonzero/video['sequence'].shape[0]*100)


'''

'''
dataset_cooc_video = []

for video in dataset_boo_video:

	n_frame = video['final_nframes']
	n_batch = 30

	video_batchedboo_matrix = np.zeros((int(n_frame/n_batch),n_feature))

	iteration = int(n_frame/n_batch)
	cooc_flat_seq_matrix = np.zeros((iteration, (n_feature-1)*(n_feature+1-1)//2), dtype=np.uint8)

	for i in range(iteration):
		frame_batch = video['sequence'][(n_batch*i):((n_batch*i)+n_batch),:]
		frame_batch = np.where(frame_batch>0,1,0)
		cooc_tri_upper = np.triu(frame_batch.T @ frame_batch, 1)

		cooc_flat_index = 0
		for j in range(n_feature-1):
			for k in range((j+1),n_feature):
				cooc_flat_seq_matrix[i, cooc_flat_index] = cooc_tri_upper[j,k]
				cooc_flat_index+=1


	dataset_cooc_video.append({'class_id': video['class_id'],
                              'final_nframes': video['final_nframes'],
                              'reduced_fps':video['reduced_fps'],
                              'sequence': cooc_flat_seq_matrix})#np.where(cooc_flat_seq_matrix>0,1,0)

'''


#============final transformation (sequence and one_hot)===========

X,y,seq_len=[],[],[]

for index,i in enumerate(dataset_batchedboo_video):
	X.append([frame_detection.tolist() for frame_detection in i['sequence']])
	one_hot = [0]*max_class_id
	one_hot[i['class_id']-1] = 1
	y.append(one_hot)
	seq_len.append(i['sequence'].shape[0])



#==========splitting==============
X_train, X_test, y_train, y_test, seq_len_train, seq_len_test = \
	 train_test_split(X,y,seq_len,test_size=0.2, random_state=0)#, stratify=y)


print('Train len %d' % len(X_train))
print('Test len %d' % len(X_test))



# =====dataset statistics=====

min_n_frame = min(seq_len)
max_n_frame = max(seq_len)

print('Full')
print(np.histogram([i['sequence'].shape[0] for i in dataset_batchedboo_video], bins=range(min_n_frame,max_n_frame+50,50)))

print('Train')
print(np.histogram(seq_len_train, bins=range(min_n_frame,max_n_frame+50,50)))

print('Test')
print(np.histogram(seq_len_test, bins=range(min_n_frame,max_n_frame+50,50)))


#-----------------------------------------------------------------------------
#------------------------------------NETWORK----------------------------------
#-----------------------------------------------------------------------------

# NN params
lstm_in_cell_units=20 # design choice (hyperparameter)

# training params
n_epoch = 100
train_batch_size=32
train_fakebatch_size = len(X_train)
test_fakebatch_size = len(X_test)
learning_rate=0.0005
#learning_rate=0.05

# ********************************************************
#!!!!IMPORTANTEEEEE!!!
# handling last batch remainder
n_iteration = len(X_train)//train_batch_size
print(n_iteration)
# *********************************************************

zipped_train_data = list(zip(X_train,y_train,seq_len_train))
zipped_test_data = list(zip(X_test,y_test,seq_len_test))



#=========================graph===========================
#tf.set_random_seed(1234)

lstmstate_batch_size = tf.placeholder(tf.int32, shape=[])

# dataset
train_data = tf.data.Dataset.from_generator(lambda: zipped_train_data, (tf.int32, tf.int32, tf.int32))
test_data = tf.data.Dataset.from_generator(lambda: zipped_test_data, (tf.int32, tf.int32, tf.int32))

# shuffle (whole) train_data
train_data = train_data.shuffle(buffer_size=len(X_train))

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
#state_c, state_h = lstm_cell.zero_state(lstmstate_batch_size, tf.float32)
#initial_state = tf.nn.rnn_cell.LSTMStateTuple(tf.Variable(state_c, trainable=False), tf.Variable(state_h, trainable=False))
initial_state = lstm_cell.zero_state(lstmstate_batch_size, tf.float32)
_, states = tf.nn.dynamic_rnn(lstm_cell, current_X_batch, initial_state=initial_state, sequence_length=current_seq_len_batch, dtype=tf.float32)


# last_step_output done right (each instance will have it's own seq_len therefore the right last ouptut for each instance must be taken)
#last_step_output = tf.gather_nd(outputs, tf.stack([tf.range(tf.shape(current_X_batch)[0]), current_seq_len_batch-1], axis=1))

# logits
#hidden_state = output per cui last_step_output è superfluo, grazie a current_seq_len_batch ritorna l'hidden_state del giusto timestep
#states è una tupla (cell_state, hidden_state) dell'ultimo timestep (in base a current_seq_len_batch)
logits = tf.layers.dense(states[1], units=max_class_id)

# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=current_y_batch))

# optimization (only during training phase (OBVIOUSLY))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# ops for accuracy and confusion matrix 
y_pred = tf.argmax(logits, 1)
y_true = tf.argmax(current_y_batch, 1)
correct_pred = tf.equal(y_pred, y_true)
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

init = tf.global_variables_initializer()

# debugging & training visualization
all_variables = tf.global_variables()

for i in all_variables:
	tf.summary.histogram(i.name.replace(':','_'), i)

summaries = tf.summary.merge_all()

losses = {
		  'train_loss':[],
		  'train_acc':[],
		  'test_loss':[],
		  'test_acc':[]
		  }


#==========================session==========================

with tf.Session() as sess:
	
	sess.run(init)	
	writer = tf.summary.FileWriter("variable_histograms")
	
	#***************** TRAINING ************************

	for i in range(n_epoch):
		writer.add_summary(sess.run(summaries), global_step=i)
		
		start_epoch_time = time.time()
		print('\nEpoch: %d/%d' % ((i+1), n_epoch))
		sess.run(train_iterator_init)
		
		for j in range(n_iteration):

			start_batch_time = time.time()
			_, batch_loss = sess.run((optimizer, loss), feed_dict={lstmstate_batch_size:train_batch_size})
			batch_time = str(datetime.timedelta(seconds=round(time.time()-start_batch_time, 2)))
			print('Batch: %d/%d - Loss: %f - Time: %s' % ((j+1), n_iteration, batch_loss, batch_time))

			# print('Batch')
			# results = sess.run((#current_X_batch, 
			# 					#current_y_batch, 
			# 					#current_seq_len_batch,
			# 					states), feed_dict={lstmstate_batch_size:train_batch_size})
			# print(results[0][1])



		
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


	sess.run(faketest_iterator_init)
	test_y_true, test_y_pred = sess.run((y_true, y_pred),feed_dict={lstmstate_batch_size:test_fakebatch_size})

	print()
	print(classid_to_classlbl)
	print()
	print(confusion_matrix(test_y_true, test_y_pred))
	print()
	print(classification_report(test_y_true, test_y_pred))
	print()
	misclassified_nframe = [seq_len[i[0]]*n_batch for i in np.argwhere(np.equal(test_y_true,test_y_pred)==False)]
	print(misclassified_nframe)

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
