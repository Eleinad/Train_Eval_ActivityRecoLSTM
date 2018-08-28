import json
import os
import cv2

'''
annotations = json.load(open('./Personal_Care/activity_net.v1-3.min.json','r'))


dataset_path = './Personal_Care'
video_folders = os.listdir(dataset_path)
video_folders = sorted([i for i in video_folders if i[0] == '_'])

dataset_video_annotations = {}

for classlbl in video_folders:
    for video in os.listdir(dataset_path+'/'+classlbl):
    	vcapture = cv2.VideoCapture(dataset_path+'/'+classlbl+'/'+video)
    	n_frame = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))
    	fps = vcapture.get(cv2.CAP_PROP_FPS)

    	curr_ann = annotations['database'][video[:video.find('.')]]['annotations']
    	for j in curr_ann:
    		segment_time = int(j['segment'][1] - j['segment'][0])
    		segment_frame = fps*segment_time
    		j['segment_frame'] = segment_frame

    	dataset_video_annotations[video[:video.find('.')]] = {'annotations': curr_ann,
										'duration': annotations['database'][video[:video.find('.')]]['duration'],
										'n_frame':n_frame,
										'fps':fps}


tot_frames = 0
tot_segm_frames = 0
for i in dataset_video_annotations.values():
	for j in i['annotations']:
		tot_segm_frames = tot_segm_frames + j['segment_frame']
	tot_frames = tot_frames + i['n_frame']



n_segm = sorted([len(i['annotations']) for i in dataset_video_annotations.values()])
'''

'''
import pickle
import matplotlib.pyplot as plt


losses = pickle.load(open('losses_nobatch.pickle','rb'))
losses_3 = pickle.load(open('losses_batched3.pickle','rb'))
losses_6 = pickle.load(open('losses_batched6.pickle','rb'))
losses_9 = pickle.load(open('losses_batched9.pickle','rb'))
losses_15 = pickle.load(open('losses_batched15.pickle','rb'))


plt.subplot(2,1,1)
plt.plot(losses['train_loss'], label='no')
plt.plot(losses['test_loss'], label='no')
plt.plot(losses_3['train_loss'], '--', color='blue', label='3')
plt.plot(losses_3['test_loss'], '--', color='orange',label='3')
plt.plot(losses_6['train_loss'], 'o', color='blue', label='6')
plt.plot(losses_6['test_loss'], 'o', color='orange',label='6')
plt.plot(losses_9['train_loss'], 'x', color='blue', label='9')
plt.plot(losses_9['test_loss'], 'x', color='orange',label='9')
plt.plot(losses_15['train_loss'], '+', color='blue', label='15')
plt.plot(losses_15['test_loss'], '+', color='orange',label='15')

plt.legend(loc='upper left')

plt.subplot(2,1,2)
plt.plot(losses['train_acc'], label='no')
plt.plot(losses['test_acc'], label='no')
plt.plot(losses_3['train_acc'], '--', color='blue',label='3')
plt.plot(losses_3['test_acc'], '--', color='orange', label='3')
plt.plot(losses_6['train_acc'], 'o', color='blue',label='6')
plt.plot(losses_6['test_acc'], 'o', color='orange',label='6')
plt.plot(losses_9['train_acc'], 'x', color='blue',label='6')
plt.plot(losses_9['test_acc'], 'x', color='orange',label='6')
plt.plot(losses_15['train_acc'], '+', color='blue',label='6')
plt.plot(losses_15['test_acc'], '+', color='orange',label='6')

plt.legend(loc='upper left')
plt.xlabel('epoch')
plt.show()
'''
import pickle
import matplotlib.pyplot as plt


losses_b9_speed= pickle.load(open('losses_b9_speed.pickle','rb'))
losses_b9_batchedboo = pickle.load(open('losses_b9_batchedboo.pickle','rb'))



plt.subplot(2,1,1)
plt.plot(losses_b9_speed['train_loss'], label='speed')
plt.plot(losses_b9_speed['test_loss'], label='speed')
plt.plot(losses_b9_batchedboo['train_loss'], '--', color='blue', label='boo')
plt.plot(losses_b9_batchedboo['test_loss'], '--', color='orange',label='boo')


plt.legend(loc='upper left')

plt.subplot(2,1,2)
plt.plot(losses_b9_speed['train_acc'], label='speed')
plt.plot(losses_b9_speed['test_acc'], label='speed')
plt.plot(losses_b9_batchedboo['train_acc'], '--', color='blue',label='boo')
plt.plot(losses_b9_batchedboo['test_acc'], '--', color='orange', label='boo')


plt.legend(loc='upper left')
plt.xlabel('epoch')
plt.show()



'''
import pickle
from pprint import pprint
from PIL import Image, ImageDraw
import numpy as np

def decode_mask(encoded_list):
    
    #DECODING    
    decoded_mask = np.zeros(encoded_list[0], dtype=bool)
    encoded_list.pop(0)
    
    for element in encoded_list:
        for i in np.arange(element[3]):
            decoded_mask[element[0]+i,element[1],element[2]]=True
    
    return decoded_mask


a = pickle.load(open('p2n_FtzA1gk_trimmed_ema.pickle','rb'))
masks = a['segments'][0]['frames_info'][50]['obj_masks']

masks = decode_mask(masks)
print(masks.shape)



img = Image.fromarray(masks[:,:,0].astype(np.uint8)*255)
#pdraw.point([find_centroid(immagine)], fill=(255,0,0))
#pdraw.rectangle(find_max_coord(x_coord, y_coord, h), outline=(255,0,0))
img.show()

#pprint(a)'''


