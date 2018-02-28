#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 14:45:15 2017

@author: ludanzhang
"""
import numpy as np
import gzip
import sys
#import time

#start = time.time()

#pass system arguments to variables
train_path = sys.argv[1]
train_path2 = sys.argv[2]
test_path = sys.argv[3]
test_path2 = sys.argv[4]

#write function to read and format data
def  data_reading(file_name,head_size,data_size):
    f = open(file_name, 'rb')
    a = gzip.GzipFile(fileobj=f) 

    buf = a.read(head_size)
    buf = a.read(data_size)
    data = np.frombuffer(buf, dtype=np.uint8)
    
    return data

#read data
train_set = data_reading(train_path,16,28*28*60000)/255.
train_label = data_reading(train_path2,8,60000)
test_set = data_reading(test_path,16,28*28*10000)/255.
test_label = data_reading(test_path2,8,10000)

#show the image to make sure the data was read in right way
#import matplotlib.pyplot as plt
#train_set_images = train_set.reshape(60000,784)
#train_set_image1 = train_set_images[1].reshape(28,28)
#plt.imshow(train_set_image1)

#reshape train ans test set into matrixs
train_set = train_set.reshape(60000,784)
test_set = test_set.reshape(10000,784)


#function to optimize w,b
#alpha as learning rate, default is 0.5
#N as iteration times, here is the number of images, default is 10000
def optimization(X,y,w,b,alpha = 0.5,N = 10000):
    
    for i in range(N):
        
            x = X[i]
            a = y[i]
        
            #compute l
            l = np.dot(x,w) + b
    
            #compute p
            p = np.exp(l).T/(np.sum(np.exp(l),axis = 1).T)
            p = p.T
    
            #compute dl
            dl = p
            dl[:,a] = p[:,a] - 1
    
            #compute dw,db
            dw = -alpha*np.dot(x.reshape(784,1),dl)
            db = -alpha*np.sum(dl,axis = 0)
    
            #refresh w,b
            w = w+dw
            b = b+db
        
    return w,b
    

#function to predict

def prediction(X,w,b):
    
    l = np.dot(X,w) + b
    
    p = np.exp(l).T/(np.sum(np.exp(l),axis = 1).T)
    p = p.T
    
    pred = np.argmax(p,axis = 1)
    
    return pred


#initialize w,b
w0 = np.zeros((784,10))
b0 = np.zeros((1,10))

#call function to train the NN 
w,b = optimization(train_set,train_label,w0,b0)
#call function to predict
pred = prediction(test_set,w,b)


#compute accuracy
accuracy = np.sum(pred == test_label)/10000.

#compute running time
#end = time.time()
#elapse = end - start

#print results
print "The accuracy of the prediction is:" , accuracy
#print ("The running time was" , elapse , "seconds." )

    
    
        
    








                             