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
                nomask_frame_info = []
                for frame in seg['frames_info']:
                    nomask_frame_info.append({'original_index':frame['original_index'],
                                                'obj_class_ids':frame['obj_class_ids'],
                                                'obj_rois':frame['obj_rois']})
                
                nomask_pic = {'video_name':pic['video_name'],
                            'class_id':pic['class_id'],
                            'seg_id':curr_seg,
                            'seg':pic['n_segments'],
                            'fps':pic['fps'],
                            'frames':len(seg['frames_info']),
                            'frames_info':nomask_frame_info}

                dataset_detection_video.append(nomask_pic)

    print(len(dataset_detection_video))


    dataset_detection_video = [video for video in dataset_detection_video if video['frames'] > 0]

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

    print('Video: %d' % len(dataset_detection_video))
    print()
    print('Activity id:')
    pprint(classlbl_to_classid)
    print()
    print('Activity distribution:')
    pprint(class_statistics)

    return dataset_detection_video, classlbl_to_classid


# bag of objects @ frame level 
def bag_of_objects(dataset_detection_video):

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

    return dataset_boo_video


# bag of objects @ batch of frame level
def batched_bag_of_objects(dataset_detection_video, batch_len):

    dataset_boo_video = bag_of_objects(dataset_detection_video)

    print('batched bag of objects')

    dataset_batchedboo_video = []

    for video in dataset_boo_video:

        n_frame = video['frames']

        if n_frame < batch_len:
            print('-> warning:short video')
            n_batch = n_frame

        n_batch = batch_len

        video_batchedboo_matrix = np.zeros((int(n_frame/n_batch),n_feature))

        iteration = int(n_frame/n_batch)

        for i in range(iteration):
            frame_batch = video['sequence'][(n_batch*i):((n_batch*i)+n_batch),:]
            video_batchedboo_matrix[i] = np.sum(frame_batch, axis=0)

        dataset_batchedboo_video.append({'video_name':video['video_name'],
                                  'class_id': video['class_id'],
                                  'frames': video['frames'],
                                  'fps':video['fps'],
                                  'sequence': video_batchedboo_matrix}) 

    return dataset_batchedboo_video


# cooccurence of objects @ batch of frame level (presence) 
def cooccurence(dataset_detection_video, batch_len):
    
    dataset_boo_video = bag_of_objects(dataset_detection_video)

    print('cooccurence')

    dataset_cooc_video = []

    for video in dataset_boo_video:

        n_frame = video['frames']

        if n_frame < batch_len:
            print('-> warning:short video')
            n_batch = n_frame

        n_batch = batch_len

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


        dataset_cooc_video.append({'video_name':video['video_name'],
                                  'class_id': video['class_id'],
                                  'frames': video['frames'],
                                  'fps':video['fps'],
                                  'sequence': cooc_flat_seq_matrix})#np.where(cooc_flat_seq_matrix>0,1,0)
    return dataset_cooc_video


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

    pickle_path = './PersonalCare/pickle'

    for video in dataset_detection_video:

        masks = []
        print(video['video_name'])
        curr_pickle = pickle.load(open(pickle_path+'/'+video['video_name'][:-4]+'_trimmed.pickle','rb'))
        for frame in curr_pickle['segments'][video['seg_id']]['frames_info']:
            masks.append(frame['obj_masks'])
		
        n_frame = video['frames']

        if n_frame < batch_len:
            print('-> warning:short video')
            n_batch = n_frame
        
        n_batch = batch_len

        interaction_frame_matrix = np.zeros( (n_frame, n_feature , n_feature) , dtype=np.uint8)

        for i in range(n_frame):
            encoded_masks = masks[i]
            decoded_mask = decode_mask(encoded_masks)

            n_current_class_ids = len(video['frames_info'][i]['obj_class_ids'])

            for j in range(n_current_class_ids):
                current_class_id_row = video['frames_info'][i]['obj_class_ids'][j]
                for k in range(n_current_class_ids):
                    current_class_id_column = video['frames_info'][i]['obj_class_ids'][k]
                    if np.sum(decoded_mask[:,:,j] & decoded_mask[:,:,k]) > 0:
                        interaction_frame_matrix[ i, current_class_id_row-1 , current_class_id_column-1 ] = 1


        iteration = int(n_frame/n_batch)
        interaction_flat_seq_matrix = np.zeros((iteration, (n_feature-1)*(n_feature+1-1)//2), dtype=np.uint8)

        for i in range(iteration):
            
            interaction_frame_matrix_batch = np.sum(interaction_frame_matrix[n_batch*i:n_batch*i+n_batch,:,:], axis=0)
            interaction_tri_upper = np.triu( interaction_frame_matrix_batch, 1)

            intersection_flat_index = 0
            for j in range(n_feature-1):
                for k in range((j+1),n_feature):
                    interaction_flat_seq_matrix[i, intersection_flat_index] = interaction_tri_upper[j,k]
                    intersection_flat_index+=1


        dataset_intersection_video.append({'video_name':video['video_name'],
                                          'class_id': video['class_id'],
										  'frames': video['frames'],
										  'fps':video['fps'],
										  'sequence': interaction_flat_seq_matrix,#np.where(cooc_flat_seq_matrix>0,1,0)
										  })
    return dataset_intersection_video

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


        img = Image.fromarray(binary_sequence.astype(np.uint8)*255)
        img.show()


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
            print('-> warning:short video')
            n_batch = n_frame
        
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

    return dataset_batchedspeed_video, dataset_batchedvelocity_video


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