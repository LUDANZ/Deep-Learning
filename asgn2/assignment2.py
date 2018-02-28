
# coding: utf-8

import tensorflow as tf
import numpy as np
import sys
import time


#import the data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

start = time.time()
#define the batch size
batchSz = 200

#input_size
input_size = mnist.train.images.shape[1]

#initialize an extra hidden layer
def hidden_layer(input_size, hidden_size=784):
    U = tf.Variable(tf.random_normal([input_size,hidden_size],stddev = 0.1, seed=10))
    bU = tf.Variable(tf.random_normal([hidden_size], stddev = 0.1, seed=10))
    return U, bU

#initialize the final layer
def final_layer(hidden_size, output_size=10):
    W = tf.Variable(tf.random_normal([hidden_size,output_size], stddev = 0.1, seed=10))
    bW = tf.Variable(tf.random_normal([output_size], stddev = 0.1, seed=10))
    return W, bW

U, bU = hidden_layer(input_size,1000)
W, bW = final_layer(1000,10)


#construct the tensor
img = tf.placeholder(tf.float32, [batchSz,input_size])
label = tf.placeholder(tf.float32, [batchSz, 10])

output1 = tf.nn.relu(tf.matmul(img, U) + bU)
logit = tf.matmul(output1, W) + bW
probs = tf.nn.softmax(tf.matmul(output1, W) + bW) 
xEnt = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label, logits = logit))

train = tf.train.GradientDescentOptimizer(0.8).minimize(xEnt)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(probs,1), tf.argmax(label,1)),
                                          tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())


#train the NN
loss = 0
numImage = mnist.train.images.shape[0]
for i in range(2000):
    imgs, labels = mnist.train.next_batch(batchSz)
    sess.run(train, feed_dict = {img: imgs, label: labels})
    loss = loss + sess.run(xEnt, feed_dict = {img:imgs, label:labels})
    if (i+1)%200 == 0:
	    print "loss after %r is:" %(i+1), loss/(i+1)
	    
#accuracy on test set
testAcc = 0
for i in range(100):
    imgs, labels = mnist.test.next_batch(batchSz)
    testAcc += sess.run(accuracy, feed_dict = {img: imgs, label: labels})
    
print "Test Accuracy is: %r" %(round(testAcc,2)), "%"
print "running time is", (time.time() - start)
sess.close()

