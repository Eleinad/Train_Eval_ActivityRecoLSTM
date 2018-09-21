import data_preprocessing
import model
import tensorflow as tf



# data loading (pickle)
dataset_detection_video, classlbl_to_classid = data_preprocessing.load_data()

#features
dataset_batchedboo_video = data_preprocessing.batched_bag_of_objects(dataset_detection_video, 9)
#dataset = data_preprocessing.kine([dataset_detection_video[0]], 9)

# splitting train & test
splitted_data = data_preprocessing.split_data(dataset_batchedboo_video)

# create the graph
model.graph(splitted_data)

#ops = [op.name for op in tf.get_default_graph().get_operations()]
#tensors = [op.values() for op in tf.get_default_graph().get_operations()]
#print(ops)
#print(tf.get_default_graph().collections)

# train & save 
model.train(splitted_data, classlbl_to_classid, 3, 32)

# restore & inference
#model.predict(splitted_data[1], splitted_data[3], splitted_data[5])