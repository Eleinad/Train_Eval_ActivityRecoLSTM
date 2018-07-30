import cv2
import os
import pickle
import numpy as np

'''
def video_dataset_stats():
    videos = []
    dataset_path = './PersonalCare'
    video_folders = os.listdir(dataset_path)
    video_folders = sorted([i for i in video_folders if i[0] == '_'])

    classlbl_to_id = {classlbl:id_ for id_,classlbl in enumerate(video_folders)}

    for classlbl in video_folders:
        for video in os.listdir(dataset_path+'/'+classlbl):
            curr_id = classlbl_to_id[classlbl]
            
            vcapture = cv2.VideoCapture(dataset_path+'/'+classlbl+'/'+video)
            n_frame = int(vcapture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = vcapture.get(cv2.CAP_PROP_FPS)
            
            videos.append({'class_id':curr_id,
                            'n_frame':n_frame,
                            'size':str(width)+'x'+str(height),
                            'fps':fps})    

    agg = {}

    for i in videos:
        if i['class_id'] not in agg:
            agg[i['class_id']]=[(i['class_id'],i['fps'],i['n_frame'])]
        else:
            agg[i['class_id']].append((i['class_id'],i['fps'],i['n_frame']))

    return videos, agg



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


print(classlbl_to_classid)

# videos must be at least 5 s long
dataset_detection_video = [i for i in dataset_detection_video if (i['final_nframes']//i['reduced_fps']) >= 5 and classid_to_classlbl[i['class_id']] != 'washingface']








#==================FEATURE BAG-OF-OBJS===============

max_class_id = 7 # y_true = activity
n_feature = 33 # bag-of-objects

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




# misuro quanto sono simili gli step in ognuna delle sequenze
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


# distribuzione del numero di oggetti per ciascuna classe (aggregata sui video)
classid_to_distr = {}
for i in dataset_boo_video:
    if classid_to_classlbl[i['class_id']] not in classid_to_distr:
        classid_to_distr[classid_to_classlbl[i['class_id']]] = np.sum(i['sequence'], axis=0)
    else:
        classid_to_distr[classid_to_classlbl[i['class_id']]] = classid_to_distr[classid_to_classlbl[i['class_id']]] + np.sum(i['sequence'], axis=0)

counts = np.zeros((33,7))
for index,i in enumerate(classid_to_distr.keys()):
    counts[:,index]= classid_to_distr[i]


for i in classid_to_distr.keys():
    classid_to_distr[i] = classid_to_distr[i]/np.sum(classid_to_distr[i]) #normalization

with open('distro.csv', 'w') as f:
    for class_lbl in classid_to_distr.keys():
        f.write(class_lbl+',')
    f.write('\n')
    for i in range(33):
        for value in classid_to_distr.values():
            f.write(str(value[i])+',')
        f.write('\n')

# tfidf
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(counts)

tfidf_importance_objs = np.argmax(tfidf.toarray(), axis=0)+1

print(tfidf_importance_objs)

classes = {'Electric Razor':1, 
            'Eye':2, 
            'Finger':3, 
            'Grabbing hands':4, 
            'Leg':5, 
            'Lenses':6, 
            'Makeup tool':7, 
            'Mouthwasher':8, 
            'Razor Pit':9, 
            'Shaving cream':10, 
            'Soap':11, 
            'Soapy hands':12, 
            'Straight Razor':13, 
            'Tap':14, 
            'Wash Basin':15,
            'backpack':16,
            'handbag':17,
            'cup':18,
            'bowl':19,
            'chair':20,
            'bed':21,
            'dining table':22,
            'toilet':23,
            'tv':24,
            'laptop':25,
            'microwave':26,
            'oven':27,
            'sink':28,
            'refrigerator':29,
            'book':30,
            'clock':31,
            'scissors':32,
            'Toothbrush':33
            }


'''

sequence = np.array([[0,0,1,1],
                     [1,0,0,1],
                     [1,1,1,1],
                     [0,1,1,0],
                     [1,1,1,1],
                     [1,0,1,1]])

sequence = np.vstack([sequence,[2,2,2,2]])
contiguous_intervals = []

for i in range(sequence.shape[1]):
    contiguous_intervals.append([])
    t_zero, t_uno = 2, 2
    for index,curr_value in enumerate(sequence[:,i]):
        t_due = t_uno
        t_uno = t_zero
        t_zero = curr_value
        if (t_due,t_uno,t_zero)==(0,1,1) or (t_due,t_uno,t_zero)==(2,1,1):
            temp=[]
            temp.append(index-1)
        elif (t_due,t_uno,t_zero)==(1,1,0) or (t_due,t_uno,t_zero)==(1,1,2):
            temp.append(index-1)
            temp.append(temp[1]-temp[0]+1)
            contiguous_intervals[i].append(list(temp))


