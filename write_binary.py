import os
import numpy as np
import tensorflow as tf
from PIL import Image
def write_binary(img_path, label_path, tfrecord_path, tfrecord_name):
    cwd = os.getcwd()
    ind = 0
    
    writer = tf.python_io.TFRecordWriter(cwd+tfrecord_path+tfrecord_name+'.tfrecord')
    data_path = cwd + img_path
    label_path = cwd + label_path
    labels = np.load(label_path)
    for img_name in os.listdir(data_path):
        img_path = data_path + img_name
        img = Image.open(img_path)
        img = img.resize((227, 227))
        img_raw = img.tobytes()
        label = labels[ind]
        example = tf.train.Example(features=tf.train.Features(feature={
            'pos': tf.train.Feature(float_list=tf.train.FloatList(value=label[0:3])),
            'angle': tf.train.Feature(float_list=tf.train.FloatList(value=label[3:6])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()

nn_word = ['train','valid','test']
for word in nn_word:
    write_binary('/dataset/'+word+'/', '/dataset/'+word+'_labels.npy', '/dataset/', word)