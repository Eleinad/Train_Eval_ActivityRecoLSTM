import data_preprocessing
import model
import tensorflow as tf
import numpy as np



# data loading (pickle)
dataset_detection_video, classlbl_to_classid = data_preprocessing.load_data()



# ====== GRID SEARCH TRAINING=========

frame_batch = [15]
lstm = [32,128]
relu = [16,32]

for i in lstm:
	for j in relu:
		for k in frame_batch:

			print(str(i)+'-'+str(j)+'-'+str(k))
			
			#features
			coo, feat_type = data_preprocessing.cooccurrence(dataset_detection_video, k)

			coint, feat_type = data_preprocessing.cointersection(dataset_detection_video, k)

			for index, video in enumerate(coint):
				video['sequence'] = np.concatenate((video['sequence'], cooc[i]['sequence']),axis=1)

			#splitting train & test
			splitted_data = data_preprocessing.split_data(coint)

			# create the graph
			model.graph(splitted_data,i,j)

			# train & save 
			model.train(splitted_data, classlbl_to_classid, 60, 32, feat_type, k)
	



'''
#========PREDICTION============

# data loading (pickle)
dataset_detection_video, classlbl_to_classid = data_preprocessing.load_data()

rnd_video_index = np.random.choice(len(dataset_detection_video),1)[0]

print(dataset_detection_video[rnd_video_index]['video_name'], dataset_detection_video[rnd_video_index]['class_id'])

#features
preprocessed_dataset, feat_type = data_preprocessing.cooccurrence([dataset_detection_video[rnd_video_index]], 15)

#splitting train & test
X,y,seq_len = data_preprocessing.network_input(preprocessed_dataset)

# restore & inference
test_y_true_lbl, test_y_pred_lbl = model.predict(X,y,seq_len, classlbl_to_classid)

model.video_pred(dataset_detection_video[rnd_video_index],classlbl_to_classid,test_y_true_lbl,test_y_pred_lbl)
'''