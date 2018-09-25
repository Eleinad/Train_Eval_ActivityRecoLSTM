import data_preprocessing
import model
import tensorflow as tf
import numpy as np



# data loading (pickle)
dataset_detection_video, classlbl_to_classid = data_preprocessing.load_data()

dataset_detection_video.sort(key=lambda x: x['class_id'])

frame_len = [int(i['frames']) for i in dataset_detection_video]

#features
dataset_preprocessed, feat_type = data_preprocessing.boo(dataset_detection_video)



'''
sequences = dataset_preprocessed[0]['sequence']
for i in range(1,len(dataset_preprocessed)):
    sequences = np.vstack((sequences,dataset_preprocessed[i]['sequence']))


voc = np.unique(sequences,axis=0)

print(sequences.shape)
print(voc.shape)
voc_dict = {tuple(simbol):index for index,simbol in enumerate(voc)}


from PIL import Image
img = []
color = {}
for i in range(voc.shape[0]):
    c = list(np.random.random(size=3) * 256)
    if tuple(c) in color:
        while tuple(c) in color:
            c = list(np.random.random(size=3) * 256)
        color[tuple(c)] = 0
    else:
        color[tuple(c)] = 0
color = list(color.keys())

maxr = 0
for video in dataset_preprocessed:
    if len(video['sequence']) > maxr:
        maxr = len(video['sequence'])
    img.append([color[voc_dict[tuple(j)]] for j in video['sequence']])

for i in range(len(img)):
    for j in range(maxr-len(img[i])):
        img[i].append((0,0,0))

img_array = np.asarray(img)
image_pil = Image.fromarray(img_array.astype('uint8'), 'RGB')

image_pil.show()
'''


#splitting train & test
splitted_data = data_preprocessing.split_data(dataset_preprocessed)

lstm = [4,8,16,32]
relu = [4,8,16,32]

for i in lstm:
	for j in relu:

		# # create the graph
		model.graph(splitted_data,i,j)

		# #ops = [op.name for op in tf.get_default_graph().get_operations()]
		# #tensors = [op.values() for op in tf.get_default_graph().get_operations()]
		# #print(ops)
		# #print(tf.get_default_graph().collections)

		# # train & save 
		model.train(splitted_data, classlbl_to_classid, 80, 32, feat_type)
		
		# # # restore & inference
		# # #model.predict(splitted_data[1], splitted_data[3], splitted_data[5])