import time
import math
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from sklearn.metrics import confusion_matrix
from datetime import timedelta
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
from sklearn.model_selection import train_test_split

labels=[]
images=[]
for i in range(10000):
    file_name = "../data_set/test_images1/anim{}.png".format(i)
    img = cv2.imread(file_name)
    images.append(img.tolist())
    labels.append(1)
for i in range(10000):
    file_name = "../data_set/test_images2/anim{}.png".format(i)
    img = cv2.imread(file_name)
    images.append(img.tolist())
    labels.append(2)
for i in range(5000):
    file_name = "../data_set/test_images3/anim{}.png".format(i)
    img = cv2.imread(file_name)
    images.append(img.tolist())
    labels.append(3)

images = np.array(images)
labels = np.array(labels)
X_train,y_train=shuffle(images,labels)



mean_u=0
sigma=0.1
learning_rate = 0.001
batch_size = 2000
training_epochs = 30
dropout=0.80
n_classes = 8

#save_file = 'model_new.ckpt'
from tensorflow.contrib.layers import flatten
def NeuralNet(x):    
    mu = 0
    sigma = 0.1
    # Layer 1: Convolutional. Input = 96x96x3. Output = 92x92x28x6.
    newconv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    newconv1_b = tf.Variable(tf.zeros(6))
    newconv1   = tf.nn.conv2d(x, newconv1_W, strides=[1, 1, 1, 1], padding='VALID') + newconv1_b

    # Activation.
    newconv1 = tf.nn.relu(newconv1)
    # Input = 92x92x6. Output = 80x80x12.
    newconv2_W=tf.Variable(tf.truncated_normal(shape=(13,13,6,12),mean=mu,stddev=sigma))
    newconv2_b=tf.Variable(tf.zeros(12))
    newconv2=tf.nn.conv2d(newconv1,newconv2_W,strides=[1,1,1,1],padding='VALID')+newconv2_b    
    # Pooling. Input = 80x80x12. Output = 40x40x12.
    newconv2 = tf.nn.max_pool(newconv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # layer 2 : input 40x40x12, and output 36x36x16
    newconv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 12, 16), mean = mu, stddev = sigma))
    newconv3_b = tf.Variable(tf.zeros(16))
    newconv3   = tf.nn.conv2d(newconv2, newconv3_W, strides=[1, 1, 1, 1], padding='VALID') + newconv3_b
    
    newconv3 = tf.nn.relu(newconv1)
    # Pooling. Input = 36x36x16. Output = 18x18x16.
    newconv3 = tf.nn.max_pool(newconv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 4: Convolutional. Output = 14x14x18.
    newconv4_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 18), mean = mu, stddev = sigma))
    newconv4_b = tf.Variable(tf.zeros(18))
    newconv4   = tf.nn.conv2d(newconv3, newconv4_W, strides=[1, 1, 1, 1], padding='VALID') + newconv4_b
    
    # Activation.
    newconv4 = tf.nn.relu(newconv4)

    # Pooling. Input = 14x14x18. Output = 7x7x18.
    newconv4 = tf.nn.max_pool(newconv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 7x7x18. Output = 882.
    newfc0 = flatten(newconv)
    newfc0 = tf.nn.dropout(newfc0,dropout)
    
    # Layer 3: Fully Connected. Input = 882. Output = 400.
    newfc1_W = tf.Variable(tf.truncated_normal(shape=(882, 400), mean = mu, stddev = sigma))
    newfc1_b = tf.Variable(tf.zeros(120))
    newfc1   = tf.matmul(newfc0, newfc1_W) + newfc1_b
    
    # Activation.
    newfc1    = tf.nn.relu(newfc1)
    #add dropout
    newfc1=tf.nn.dropout(newfc1,dropout)

    # Layer 4: Fully Connected. Input = 400. Output = 84.
    newfc2_W  = tf.Variable(tf.truncated_normal(shape=(400, 84), mean = mu, stddev = sigma))
    newfc2_b  = tf.Variable(tf.zeros(84))
    newfc2    = tf.matmul(newfc1, newfc2_W) + newfc2_b
    
    # Activation.
    newfc2    = tf.nn.relu(newfc2)
    #add dropout 
    newfc2 = tf.nn.dropout(newfc2, dropout)
    
    
    # Layer 5: Fully Connected. Input = 84. Output = 43.
    newfc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 8), mean = mu, stddev = sigma))
    newfc3_b  = tf.Variable(tf.zeros(8))
    logits = tf.matmul(newfc2, newfc3_W) + newfc3_b
    
    return logits


#construct functions for the neural nets
x=tf.placeholder(tf.float32,(None,96,96,3))
y=tf.placeholder(tf.int32,(None))
one_hot_y=tf.one_hot(y,8)
logits=NeuralNet(x)
cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits,one_hot_y)
loss_operation=tf.reduce_mean(cross_entropy)
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation=optimizer.minimize(loss_operation)




cross_prediction=tf.equal(tf.argmax(logits,1),tf.argmax(one_hot_y,1))
accuracy_operation=tf.reduce_mean(tf.cast(cross_prediction,tf.float32))

def evaluation(X_data,y_data):
    num_examples=len(X_data)
    total_accuracy=0
    sess=tf.get_default_session()
    for offset in range(0,num_examples,batch_size):
        batch_x,batch_y=X_data[offset:offset+batch_size],y_data[offset:offset+batch_size]
        accuracy=sess.run(accuracy_operation,feed_dict={x:batch_x,y:batch_y})
        total_accuracy+=(accuracy*len(batch_x))
    return total_accuracy/num_examples


training_data,validation_data,training_label,validation_label=train_test_split(X_train,y_train,test_size=0.2)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples=len(training_data)
    print("we are training our model")
    print()
    for i in range(training_epochs):
        X_train1,y_train1=shuffle(training_data,training_label)
        for offset in range(0,num_examples,batch_size):
            end=offset+batch_size
            batch_x,batch_y=X_train[offset:end],y_train[offset:end]
            sess.run(training_operation,feed_dict={x:batch_x,y:batch_y})
        v_accuracy=evaluation(validation_data,validation_label)
        print("epoch{}:".format(i+1))
        print("the validation accuracy:{:.3f}".format(v_accuracy))
        print()
    cross=tf.equal(tf.argmax(logits,1),tf.argmax(one_hot_y,1))
    accuracy=tf.reduce_mean(tf.cast(cross_prediction,tf.float32))   
    print("the test accuracy after using regularization is:",accuracy.eval({x:X_test,y:y_test}))
    #saver.save(sess, save_file)

