import pickle
import numpy as np
import os



def decode_mask(encoded_list):
    
    decoded_mask = np.zeros(encoded_list[0], dtype=bool)
    encoded_list.pop(0)
    
    for element in encoded_list:
        for i in np.arange(element[3]):
            decoded_mask[element[0]+i,element[1],element[2]]=True
    
    return decoded_mask

pickle_path = './PersonalCare/pickle'
light_pickle_path = './PersonalCare/light_pickle'

for video_pickle in [i for i in os.listdir(pickle_path) if 'trimmed' in i]:
    if 'trimmed' in video_pickle:
        pic = pickle.load(open(pickle_path+'/'+video_pickle,'rb'))
        print(pic['video_name'])
        for curr_seg,seg in pic['segments'].items():
            n_feature=33
            n_frame=len(seg['frames_info'])
            print(n_frame)


            for frame in seg['frames_info']:

                interaction_frame_matrix = np.zeros( (n_feature , n_feature) , dtype=np.uint8)

                encoded_masks = frame['obj_masks']
                decoded_masks = decode_mask(encoded_masks)

                n_current_class_ids = len(frame['obj_class_ids'])

                for j in range(n_current_class_ids):
                    current_class_id_row = frame['obj_class_ids'][j]
                    for k in range(n_current_class_ids):
                        current_class_id_column = frame['obj_class_ids'][k]
                        if j!=k:
                            if np.sum(np.logical_and(decoded_masks[:,:,j],decoded_masks[:,:,k])) > 0:
                                interaction_frame_matrix[current_class_id_row-1, current_class_id_column-1] = 1

                interaction_flat_seq_matrix = np.zeros(((n_feature-1)*(n_feature+1-1)//2), dtype=np.uint8)
                interaction_tri_upper = np.triu( interaction_frame_matrix, 1)

                intersection_flat_index = 0
                for j in range(n_feature-1):
                    for k in range((j+1),n_feature):
                        interaction_flat_seq_matrix[intersection_flat_index] = interaction_tri_upper[j,k]
                        intersection_flat_index+=1

                frame['obj_masks'] = interaction_flat_seq_matrix

        pickle.dump(pic, open(light_pickle_path+'/'+pic['video_name'][:-4]+'_trimmed.pickle','wb'))


