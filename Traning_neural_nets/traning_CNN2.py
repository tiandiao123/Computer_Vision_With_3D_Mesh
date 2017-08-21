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

#layer 1 whcihsaver = tf.train.Saver is 32*32*3 as input and 28*28*12 as output
conv1_w=tf.Variable(tf.truncated_normal(shape=(5,5,3,12),mean=mean_u,stddev=sigma))
conv1_b=tf.Variable(tf.zeros(12))

#layer2 which is 14*14*12 as input, and the output is 10*10*16 as output
conv2_w=tf.Variable(tf.truncated_normal(shape=(5,5,12,16),mean=mean_u,stddev=sigma))
conv2_b=tf.Variable(tf.zeros(16))

#layer 3:
fun1_w=tf.Variable(tf.truncated_normal(shape=(400,120),mean=mean_u,stddev=sigma))
fun1_b=tf.Variable(tf.zeros(120))

    
#layer 4
fun2_w=tf.Variable(tf.truncated_normal(shape=(120,84),mean=mean_u,stddev=sigma))
fun2_b=tf.Variable(tf.zeros(84))

#layrer 5
fun3_w=tf.Variable(tf.truncated_normal(shape=(84,43),mean=mean_u,stddev=sigma))
fun3_b=tf.Variable(tf.zeros(43))

def Neuralnets(X):
    #layer 1 whcih is 32*32*3 as input and 28*28*12 as output
    conv1=tf.nn.conv2d(X,conv1_w,strides=[1,1,1,1],padding='VALID')+conv1_b
    
    #apply activation:
    conv1=tf.nn.relu(conv1)
    
    #apply max_pooling: output is 14*14*12
    conv1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    
    #layer2 which is 14*14*12 as input, and the output is 10*10*16 as output
    conv2=tf.nn.conv2d(conv1,conv2_w,strides=[1,1,1,1],padding='VALID')+conv2_b
    conv2=tf.nn.relu(conv2)
    #apply max pooling and we get 5*5*16 output
    conv2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    
    #flatten we get 400
    fun0=flatten(conv2)
    
    #layer 3:
    fun1=tf.matmul(fun0,fun1_w)+fun1_b
    
    #apply actiovation: output 120
    fun1=tf.nn.relu(fun1)
    
    #layer 4
    fun2=tf.matmul(fun1,fun2_w)+fun2_b
    
    #apply actiovation:
    fun2=tf.nn.relu(fun2)
    
    logits=tf.matmul(fun2,fun3_w)+fun3_b
    return logits



#construct functions for the neural nets
x=tf.placeholder(tf.float32,(None,32,32,3))
y=tf.placeholder(tf.int32,(None))
one_hot_y=tf.one_hot(y,3)
logits=Neuralnets(x)
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

