"""Converts Gauge data to TFRecords file format with Example protos."""
import os
import tensorflow as tf
import cv2
import glob
import numpy as np
import pickle
from itertools import chain
from sklearn.model_selection import train_test_split


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name, directory):
    """Converts a dataset to tfrecords."""
    images = data_set['images']
    labels = data_set['labels']
    num_examples = data_set['num_examples']

    if images.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' %
                         (images.shape[0], num_examples))
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


'''
########## TRAINING DATASET ####################################################
'''
dim = (28, 28)
TRAIN_DIR = "/Users/mehdi/Desktop/Projects/DeepGauge/ImageEveryUnit"
ALLFOLDERS = os.listdir(TRAIN_DIR)
Folders = [x for x in ALLFOLDERS if x.startswith('psi')]
Folder_Dict = dict([(y,x) for x,y in enumerate(sorted(set(Folders)))])

# Store data (Folder dictionary)
with open('./data/label_dict.pickle', 'wb') as handle:
    pickle.dump(Folder_Dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


images = []
images_labels = []

for folder, label in Folder_Dict.items():
    # if label<10:
    print('folder = {}, label = {:d}'.format(folder, label))
    folder_dir = os.path.join(TRAIN_DIR, folder)
    files1 = glob.glob(os.path.join(folder_dir, "*.png"))  # your image path
    files2 = glob.glob(os.path.join(folder_dir, "*.PNG"))  # your image path
    files3 = glob.glob(os.path.join(folder_dir, "*.jpg"))  # your image path
    files4 = glob.glob(os.path.join(folder_dir, "*.JPG"))
    files5 = glob.glob(os.path.join(folder_dir, "*.jpeg"))  # your image path
    files6 = glob.glob(os.path.join(folder_dir, "*.JPEG"))
    files = list(chain(files1, files2, files3, files4, files5, files6))
    for myFile in files:
        image = cv2.imread (myFile)
        if image.shape[0:2] != dim:
            image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        images.append(image)
        images_labels.append(label)

train, test, train_labels, test_labels = train_test_split(images, images_labels,
                                                        test_size=0.20, random_state=100)

'''
#################### saving train dataset to tfrecords ####################
'''
train = np.array(train, dtype='uint8') #as mnist
train = train[:, :, :, 1][:, :, :,np.newaxis]
train_labels = np.array(train_labels,dtype='uint8') #as mnist
nexamples_train = len(train_labels)
train_dataset = {'images': train, 'labels': train_labels, 'num_examples': nexamples_train}
convert_to(train_dataset, 'train', 'data')

# Store data (train_dataset)
with open('./data/train_dataset.pickle', 'wb') as handle:
    pickle.dump(train_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
#################### saving test dataset to tfrecords ####################
'''
test = np.array(test, dtype='uint8') #as mnist
test = test[:, :, :, 1][:, :, :,np.newaxis]
test_labels = np.array(test_labels,dtype='uint8') #as mnist
nexamples_test = len(test_labels)
test_dataset = {'images': test, 'labels': test_labels, 'num_examples': nexamples_test}
convert_to(test_dataset, 'test', 'data')

# Store data (test_dataset)
with open('./data/test_dataset.pickle', 'wb') as handle:
    pickle.dump(test_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)



# # Load data (deserialize)
# with open('./data/test_dataset.pickle', 'rb') as handle:
#     unserialized_data = pickle.load(handle)

# for example in tf.python_io.tf_record_iterator("./data/test.tfrecords"):
#     print(tf.train.Example.FromString(example))
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
#
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# data = mnist.test.images[0].tolist()
#
# mnist.test.images[0].shape
# ((test_dataset['images'][0])/255).reshape([28*28, ]).tolist().shape