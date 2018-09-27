import data_preprocessing
import model
import tensorflow as tf
import numpy as np



# data loading (pickle)
dataset_detection_video, classlbl_to_classid = data_preprocessing.load_data()

# import matplotlib.pyplot as plt
# classid_to_classlbl = {value:key for key,value in classlbl_to_classid.items()}
# classes = [i['class_id'] for i in dataset_detection_video]
# labels, counts = np.unique(classes, return_counts=True)
# plt.bar(labels, counts, align='center', tick_label=[classid_to_classlbl[i] if 'lenses' not in classid_to_classlbl[i] else 'putcontactlenses' for i in range(7)])
# plt.gca().set_xticks(labels)
# plt.xlabel('Activity')
# plt.ylabel('Frequency')
# plt.show()

frame_len = [i['frames'] for i in dataset_detection_video]
frame_len.sort()



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


frame_batch = [15]
lstm = [16]
relu = [0]

for i in lstm:
	for j in relu:
		for k in frame_batch:

			print(str(i)+'-'+str(j)+'-'+str(k))
			#features
			speed, velocity, feat_type = data_preprocessing.kine([dataset_detection_video[1]], k)

			#splitting train & test
			splitted_data = data_preprocessing.split_data(speed)

			# create the graph
			model.graph(splitted_data,i,j)

			# train & save 
			model.train(splitted_data, classlbl_to_classid, 60, 32, feat_type, k)
			
			# restore & inference
			#model.predict(splitted_data[1], splitted_data[3], splitted_data[5], classlbl_to_classid)

