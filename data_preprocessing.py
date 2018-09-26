import pickle
import os
from pprint import pprint
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image




#============data parameters==========

max_class_id = 7 # y_true = activity
n_feature = 33 # bag-of-objects


# loading data from serialized .pickle files
def load_data():

    dataset_detection_video = []

    pickle_path = './PersonalCare/pickle'

    for video_pickle in os.listdir(pickle_path):
        if 'trimmed' in video_pickle:
            pic = pickle.load(open(pickle_path+'/'+video_pickle,'rb'))
            for curr_seg,seg in pic['segments'].items():

                mod_pic = {'video_name':pic['video_name'],
                            'class_id':pic['class_id'],
                            'seg_id':curr_seg,
                            'segs':pic['n_segments'],
                            'fps':pic['fps'],
                            'frames':len(seg['frames_info']),
                            'frames_info':seg['frames_info']}

                dataset_detection_video.append(mod_pic)


    dataset_detection_video = [video for video in dataset_detection_video if video['frames'] > 0 or video['fps'] > 0]
    dataset_detection_video = [video for video in dataset_detection_video if int(video['frames']/video['fps']) >= 5] #at least 5 sec

    # for video in dataset_detection_video:
    #     if video['fps'] >= 29:
    #         new_frames = []
    #         for i in range(0,len(video['frames_info']),2):
    #             new_frames.append(video['frames_info'][i])

    #         video['frames_info'] = new_frames
    #         video['frames'] = len(new_frames)


    classlbl_to_classid = {}
    classid = 0

    for i in dataset_detection_video:
        classlbl = i['class_id'].lower().replace(' ','')
        if classlbl not in classlbl_to_classid:
            classlbl_to_classid[classlbl] = classid
            classid += 1
        i['class_id'] = classlbl_to_classid[classlbl]


    classid_to_classlbl = {value:key for key,value in classlbl_to_classid.items()}

    # filtering data -> videos must be at least 5 s
    #dataset_detection_video = [i for i in dataset_detection_video if (i['final_nframes']//i['reduced_fps']) >= 5]

    # classes distribution
    class_statistics = {}

    for video in dataset_detection_video:
        if classid_to_classlbl[video['class_id']] not in class_statistics:
            class_statistics[classid_to_classlbl[video['class_id']]] = 1
        else:
            class_statistics[classid_to_classlbl[video['class_id']]] += 1

    for activity in class_statistics.keys():
        class_statistics[activity] = (class_statistics[activity], round(class_statistics[activity]*100/len(dataset_detection_video)))

    print('Video: %d' % len(os.listdir(pickle_path)))
    print()
    print('Video (length filtered segs): %d' % len(dataset_detection_video))
    print()
    print('Activity id:')
    pprint(classlbl_to_classid)
    print()
    print('Activity distribution:')
    pprint(class_statistics)
    print()

    dataset_detection_video.sort(key=lambda x: x['class_id'])

    return dataset_detection_video, classlbl_to_classid


# bag of objects @ frame level 
def boo(dataset_detection_video):

    print('\nbag of objects')
    
    dataset_boo_video = []

    for video in dataset_detection_video:
        
        video_boo_matrix = np.zeros((video['frames'],n_feature), dtype=np.uint8)

        for index, frame in enumerate(video['frames_info']) :
            boo = {}

            for obj in frame['obj_class_ids']:
                if obj not in boo:
                    boo[obj] = 1
                else:
                    boo[obj] += 1

            for class_id_index, obj_freq in boo.items():
                video_boo_matrix[index][class_id_index-1] = obj_freq

          
        dataset_boo_video.append({'video_name':video['video_name'],
                                  'class_id': video['class_id'],
                                  'frames': video['frames'],
                                  'fps':video['fps'],
                                  'sequence': video_boo_matrix})

    return dataset_boo_video, 'boo'


# bag of objects @ batch of frame level
def batched_boo(dataset_detection_video, batch_len):

    dataset_boo_video = boo(dataset_detection_video)

    print('batched bag of objects')

    dataset_batchedboo_video = []

    for video in dataset_boo_video[0]:

        n_frame = video['frames']

        if n_frame < batch_len:
            print('-> warning:short batch')
            n_batch = n_frame
        else:
            n_batch = batch_len

        video_batchedboo_matrix = np.zeros((int(n_frame/n_batch),n_feature))

        iteration = int(n_frame/n_batch)

        for i in range(iteration):
            frame_batch = video['sequence'][(n_batch*i):((n_batch*i)+n_batch),:]
            video_batchedboo_matrix[i] = np.sum(frame_batch, axis=0)/n_batch

        dataset_batchedboo_video.append({'video_name':video['video_name'],
                                  'class_id': video['class_id'],
                                  'frames': video['frames'],
                                  'fps':video['fps'],
                                  'sequence': video_batchedboo_matrix}) 

    return dataset_batchedboo_video, 'bboo'


# cooccurence of objects @ batch of frame level (presence) 
def cooccurrence(dataset_detection_video, batch_len):
    
    dataset_boo_video = boo(dataset_detection_video)

    print('cooccurence')

    dataset_cooc_video = []

    for video in dataset_boo_video[0]:

        n_frame = video['frames']

        if n_frame < batch_len:
            print('-> warning:short batch')
            n_batch = n_frame
        else:
            n_batch = batch_len

        video_batchedboo_matrix = np.zeros((int(n_frame/n_batch),n_feature))

        iteration = int(n_frame/n_batch)
        cooc_flat_seq_matrix = np.zeros((iteration, (n_feature)*(n_feature+1)//2), dtype=np.uint8)

        for i in range(iteration):
            frame_batch = video['sequence'][(n_batch*i):((n_batch*i)+n_batch),:]
            frame_batch = np.where(frame_batch>0,1,0)
            cooc_tri_upper = np.triu(frame_batch.T @ frame_batch, 0)

            cooc_flat_index = 0
            for j in range(n_feature):
                for k in range(j,n_feature):
                    cooc_flat_seq_matrix[i, cooc_flat_index] = cooc_tri_upper[j,k]
                    cooc_flat_index+=1


        dataset_cooc_video.append({'video_name':video['video_name'],
                                  'class_id': video['class_id'],
                                  'frames': video['frames'],
                                  'fps':video['fps'],
                                  'sequence': cooc_flat_seq_matrix})#np.where(cooc_flat_seq_matrix>0,1,0)
    
    return dataset_cooc_video, 'cooc'


def decode_mask(encoded_list):
    
    decoded_mask = np.zeros(encoded_list[0], dtype=bool)
    encoded_list.pop(0)
    
    for element in encoded_list:
        for i in np.arange(element[3]):
            decoded_mask[element[0]+i,element[1],element[2]]=True
    
    return decoded_mask


# intersection of masks @ batch of frame level (presence) 
def cointersection(dataset_detection_video, batch_len):
    
    print('cointersection')

    dataset_intersection_video = []

    for video in dataset_detection_video:

        n_frame = video['frames']

        if n_frame < batch_len:
            print('-> warning:short batch')
            n_batch = n_frame
        else:
            n_batch = batch_len
        
        interaction_flat_matrix = [frame['obj_masks'] for frame in video['frames_info']]
        interaction_flat_matrix = np.asarray(interaction_flat_matrix)

        iteration = int(n_frame/n_batch)
        interaction_flat_batch_matrix = np.zeros((iteration, (n_feature-1)*(n_feature+1-1)//2), dtype=np.uint8)

        for i in range(iteration):
            interaction_flat_batch_matrix[i] = np.sum(interaction_flat_matrix[n_batch*i:n_batch*i+n_batch,:], axis=0)


        dataset_intersection_video.append({'video_name':video['video_name'],
                                          'class_id': video['class_id'],
										  'frames': video['frames'],
										  'fps':video['fps'],
										  'sequence': interaction_flat_batch_matrix,#np.where(cooc_flat_seq_matrix>0,1,0)
										  })
    
    return dataset_intersection_video, 'coint'

# speed and velocity of object centroids @ batch of frame level based on object contiguous 
def kine(dataset_detection_video, batch_len):

    print('speed & velocity')
    
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

        # for frame in video['frames_info']:
        #     centroids_list.append([[] for _ in range(33)])
        #     objs = frame['obj_class_ids']
        #     rois = frame['obj_rois']
        #     for i in range(objs.shape[0]):
        #         curr_obj_roi = rois[i]
        #         curr_obj_id = objs[i]-1
        #         (x, y) = centroid_roi(curr_obj_roi)
        #         centroids_list[-1][curr_obj_id].append((int(x),int(y))

        n_frames = video['frames']
        start_frame_index = video['frames_info'][0]['original_index']
        
        i,k=0,0
        curr_frame_index = start_frame_index
        while curr_frame_index<start_frame_index+n_frames:
            centroids_list.append([[] for _ in range(33)])
            #centroids_list_mask.append([[] for _ in range(33)])
            if video['frames_info'][i]['original_index'] == curr_frame_index:
                objs = video['frames_info'][i]['obj_class_ids']
                #masks = decode_mask(a['segments'][0]['frames_info'][i]['obj_masks'])
                rois = video['frames_info'][i]['obj_rois']
                for j in range(objs.shape[0]):
                    #curr_obj_mask = Image.fromarray(masks[:,:,j].astype(np.uint8)*255)
                    #curr_obj_mask = masks[:,:,j]
                    curr_obj_roi = rois[j]
                    curr_obj_id = objs[j]-1
                    (x_roi, y_roi) = centroid_roi(curr_obj_roi)
                    #(x_mask, y_mask) = centroid_mask(curr_obj_mask)
                    centroids_list[k][curr_obj_id].append((int(x_roi),int(y_roi)))
                    #centroids_list_mask[k][curr_obj_id].append((int(x_mask),int(y_mask)))
                    
                    #pdraw = ImageDraw.Draw(curr_obj_mask)
                    #pdraw.point([centroid_mask(curr_obj_mask)], fill=125)
                    #curr_obj_mask.show()
                i+=1
            curr_frame_index+=1
            k+=1




        # encoding di centroids_list in una binary matrix
        # da usare dopo per ottenere objid_to_contiguous_intervals
        n = video['frames']

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
        #    .keys  = objid (int)
        #    .value = start, end, length degli intervalli contigui (list of lists)
        # 2. objid_to_listavgspeedincontiguous (dict)
        #    .keys  = objid (int)
        #    .value = speed nel corrispettivo contiguo
        # sfruttando queste due vengono costruite le features finali


        n_frame = video['frames']

        if n_frame < batch_len:
            print('-> warning:short batch')
            n_batch = n_frame
        else:
            n_batch = batch_len

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

            for objid, values in temp.items():
                video_batchedspeed_matrix[i][objid-1] = values[1]
                video_batchedvelocity_matrix[i][objid-1] = values[0][0]
                video_batchedvelocity_matrix[i][objid] = values[0][1]

        
        dataset_batchedspeed_video.append({'video_name':video['video_name'],
                                  'class_id': video['class_id'],
                                  'frames': video['frames'],
                                  'fps':video['fps'],
                                  'sequence': video_batchedspeed_matrix})
        dataset_batchedvelocity_video.append({'video_name':video['video_name'],
                                  'class_id': video['class_id'],
                                  'frames': video['frames'],
                                  'fps':video['fps'],
                                  'sequence': video_batchedvelocity_matrix})

    return dataset_batchedspeed_video, dataset_batchedvelocity_video, 'kine'


def split_data(dataset):
    
    X,y,seq_len=[],[],[]

    for index,i in enumerate(dataset):
        X.append([frame_detection.tolist() for frame_detection in i['sequence']])
        one_hot = [0]*max_class_id
        one_hot[i['class_id']-1] = 1
        y.append(one_hot)
        seq_len.append(i['sequence'].shape[0])


    X_train, X_test, y_train, y_test, seq_len_train, seq_len_test = \
         train_test_split(X,y,seq_len,test_size=0.2, random_state=0)#, stratify=y)

    print()
    print('Train len: %d' % len(X_train))
    print('Test len: %d' % len(X_test))

    return X_train, X_test, y_train, y_test, seq_len_train, seq_len_test