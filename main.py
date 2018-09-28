import data_preprocessing
import model
import tensorflow as tf
import numpy as np



# data loading (pickle)
dataset_detection_video, classlbl_to_classid = data_preprocessing.load_data()



# ====== GRID SEARCH TRAINING=========

frame_batch = [15]
lstm = [16]
relu = [0]

for i in lstm:
	for j in relu:
		for k in frame_batch:

			print(str(i)+'-'+str(j)+'-'+str(k))
			
			#features
			pre, feat_type = data_preprocessing.cooc(dataset_detection_video, k)

			#splitting train & test
			splitted_data = data_preprocessing.split_data(pre)

			# create the graph
			model.graph(splitted_data,i,j)

			# train & save 
			model.train(splitted_data, classlbl_to_classid, 35, 32, feat_type, k)
			



'''
#========PREDICTION============

# data loading (pickle)
dataset_detection_video, classlbl_to_classid = data_preprocessing.load_data()

#features
preprocessed_dataset, feat_type = data_preprocessing.cooc(dataset_detection_video, 15)

#splitting train & test
splitted_data = data_preprocessing.split_data(preprocessed_dataset)

X_test = splitted_data[1]
y_test = splitted_data[3]
seq_len_test = splitted_data[5]


rnd_video_index = np.random.choice(len(X_test),1)[0]

# restore & inference
model.predict(splitted_data[1][rnd_video_index], 
			   splitted_data[3][rnd_video_index], 
			   splitted_data[5][rnd_video_index], classlbl_to_classid)
'''