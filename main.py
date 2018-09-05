import data_preprocessing
import model
import tensorflow as tf



dataset_detection_video, classlbl_to_classid = data_preprocessing.load_data()

dataset_batchedboo_video = data_preprocessing.batched_bag_of_objects(dataset_detection_video, 9)

splitted_data = data_preprocessing.split_data(dataset_batchedboo_video)

model.graph(splitted_data)

ops = [op.name for op in tf.get_default_graph().get_operations()]
tensors = [op.values() for op in tf.get_default_graph().get_operations()]

#print(tensors)

#print(tf.get_default_graph().collections)

model.train(splitted_data, classlbl_to_classid, 10, 32)