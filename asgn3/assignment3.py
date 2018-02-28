
# coding: utf-8


import tensorflow as tf
import numpy as np

#import the data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#define hyper parameter
batchSz = 50
iteration = 2000
learning_rate = 0.0001

#placeholder
img = tf.placeholder(dtype=tf.float32, shape=[None,784])
label = tf.placeholder(dtype=tf.float32, shape=[None,10])
img_ipt = tf.reshape(img, shape=[-1,28,28,1])

#convlution layer
flt1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev = 0.1))
convOut1 = tf.nn.conv2d(img_ipt, flt1, [1,1,1,1], "SAME")
convOut1 = tf.nn.relu(convOut1)
convOut1 = tf.nn.max_pool(convOut1,[1,2,2,1],[1,2,2,1], "SAME")
    
flt2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev = 0.1))
convOut2 = tf.nn.conv2d(convOut1, flt2, [1,1,1,1], "SAME")
convOut2 = tf.nn.relu(convOut2)
convOut2 = tf.nn.max_pool(convOut2,[1,2,2,1],[1,2,2,1], "SAME")
    
convOut2 = tf.reshape(convOut2,[-1,3136])

    
#feed-forward layer
V1 = tf.Variable(tf.truncated_normal(dtype=tf.float32,shape=[3136,512],stddev=0.1))
bV1 = tf.Variable(tf.truncated_normal(dtype=tf.float32,shape=[512],stddev=0.1))
logit1 = tf.nn.relu(tf.matmul(convOut2,V1) + bV1)

V2 = tf.Variable(tf.truncated_normal(dtype=tf.float32,shape=[512,10],stddev=0.1))
bV2 = tf.Variable(tf.truncated_normal(dtype=tf.float32,shape=[10],stddev=0.1))
logit2 = tf.matmul(logit1,V2) + bV2

#softmax
probs = tf.nn.softmax(logit2)
xEnt = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit2,labels=label))

#train
train = tf.train.AdamOptimizer(learning_rate).minimize(xEnt)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(probs,1), tf.argmax(label,1)),
                                          tf.float32))
#create session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#train NN
for i in range(iteration):
    imgs, labels = mnist.train.next_batch(batchSz)
    sess.run(train, feed_dict = {img: imgs, label: labels})

#test accuracy
testAcc = 0
for i in range(1000):
    imgs, labels = mnist.test.next_batch(batchSz)
    testAcc += sess.run(accuracy, feed_dict = {img: imgs, label: labels})
    
print "Test Accuracy is: %r" %(testAcc/1000)


sess.close()



