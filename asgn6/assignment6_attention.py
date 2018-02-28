
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import sys
import re


# In[2]:


#preprocess
#training set
fr_train_path = sys.argv[1]
en_train_path = sys.argv[2]
def preprocess(path):
    with open(path,"r") as train:
        train_words = []
        for sentence in train.readlines():
            n = 12 - len(sentence.split())
            train_words.extend(sentence.split())
            train_words.extend(["<stop>"]*n)
    
    train_dict = {w:i for i,w in enumerate(set(train_words))}
    train_token = [train_dict[word] for word in train_words]
    train_token = np.reshape(np.array(train_token),[-1,12])
    return train_dict,train_token

fr_dict, fr_train_token = preprocess(fr_train_path)
en_dict, en_train_token = preprocess(en_train_path)
#add <start> 
en_dict["<start>"] = max(en_dict.values()) + 1
en_train_token = np.column_stack((np.array([en_dict["<start>"]]*np.shape(en_train_token)[0]),en_train_token))
#testing set
fr_test_path = sys.argv[3]
en_test_path = sys.argv[4]
def test_preprocess(path,word_dict):
    with open(path,"r") as test:
        test_words = []
        for sentence in test.readlines():
            n = 12 - len(sentence.split())
            test_words.extend(sentence.split())
            test_words.extend(["<stop>"]*n)
        
        test_token = []
        for word in test_words:
            if word in word_dict:
                test_token.append(word_dict[word])
            else:
                test_token.append(word_dict["*UNK*"])
        test_token = np.reshape(np.array(test_token),[-1,12])
        return test_token

fr_test_token = test_preprocess(fr_test_path,fr_dict)
en_test_token = test_preprocess(en_test_path,en_dict)
#add <start> 
en_test_token = np.column_stack((np.array([en_dict["<start>"]]*np.shape(en_test_token)[0]),en_test_token))


# In[3]:


#parmeter
bSz = 20
learning_rate = 1e-3
rnnSz = 64
wSz = 12
vfrSz = len(fr_dict.values())
venSz = len(en_dict.values())
embedSz = 30


# In[4]:


#construct NN
#placeholders
encIn = tf.placeholder(dtype = tf.int64, shape = [bSz,wSz])
decIn = tf.placeholder(dtype = tf.int64, shape = [bSz,wSz])
ans = tf.placeholder(dtype = tf.int64, shape = [bSz, wSz])
keepPrb = tf.placeholder(dtype = tf.float32)
maskLen = tf.placeholder(dtype = tf.int64, shape = [bSz])
frmaskLen = tf.placeholder(dtype = tf.int64, shape = [bSz])
enmaskLen = tf.placeholder(dtype = tf.int64, shape = [bSz])
# In[5]:


#encoding
with tf.variable_scope("enc"):
    F = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[vfrSz, embedSz], stddev = 0.1))
    embs = tf.nn.embedding_lookup(F,encIn)
    embs = tf.nn.dropout(embs, keepPrb)
    cell = tf.contrib.rnn.GRUCell(rnnSz)
    initialSt1 = cell.zero_state(bSz, tf.float32)
    encOut, encSt = tf.nn.dynamic_rnn(cell, embs, initial_state=initialSt1, sequence_length = frmaskLen)
    encOutT = tf.transpose(encOut,[0,2,1])
    AW0 = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[wSz,wSz], stddev = 0.1))
    AW = tf.divide(AW0,tf.reduce_sum(AW0,0))
    encOutT = tf.tensordot(encOutT, AW, [[2],[0]])
    encOut = tf.transpose(encOutT,[0,2,1])
#decoding
with tf.variable_scope("dec"):
    E = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[venSz, embedSz], stddev = 0.1))
    embs = tf.nn.embedding_lookup(E,decIn)
    embs = tf.nn.dropout(embs, keepPrb)
    inpt = tf.concat(values=[embs, encOut], axis=2)
    cell = tf.contrib.rnn.GRUCell(rnnSz)
    initialSt = cell.zero_state(bSz, tf.float32)
    decOut, decSt = tf.nn.dynamic_rnn(cell, inpt, initial_state=initialSt, sequence_length = enmaskLen)
    
#loss
W = tf.Variable(tf.truncated_normal(dtype = tf.float32, shape = [rnnSz,venSz], stddev = 0.1))
b = tf.Variable(tf.truncated_normal(dtype = tf.float32, shape = [venSz], stddev = 0.1))
logits = tf.tensordot(decOut, W, axes = [[2],[0]]) + b
prob = tf.nn.softmax(logits)
loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=ans, weights=tf.sequence_mask(maskLen,wSz, tf.float32))
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

#accuracy
result = tf.argmax(prob, axis=2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[6]:
def sequenceLen(inpt, word_dict, n):
    Len = []
    for row in range(n):
        l = np.where(inpt[row,:] == word_dict["<stop>"])[0]
	if len(l) == 0:
            Len.append(12)
        else:
	    Len.append(l[0] + 1)
    return Len

#train
nround = np.shape(en_train_token)[0]//bSz
losses = []
newSt1 = sess.run(initialSt1)
newSt = sess.run(initialSt)
for i in range(nround):
    fr_input = fr_train_token[i*bSz:(i+1)*bSz,0:wSz]
    en_input = en_train_token[i*bSz:(i+1)*bSz,0:wSz]
    en_ans = en_train_token[i*bSz:(i+1)*bSz,1:wSz+1]
    mask_len_fr = sequenceLen(fr_input, fr_dict, bSz)
    mask_len_en = sequenceLen(en_input, en_dict, bSz)
    mask_len = sequenceLen(en_ans, en_dict, bSz)
    
    sess.run(train, feed_dict = {encIn:fr_input, decIn:en_input, ans:en_ans, keepPrb:1.0, initialSt:newSt, initialSt1:newSt1, maskLen:mask_len, frmaskLen: mask_len_fr, enmaskLen: mask_len_en})
    losses_new, newSt1, newSt = sess.run([loss, encSt, decSt], feed_dict = {encIn:fr_input, decIn:en_input, ans:en_ans, keepPrb:1.0, initialSt1:newSt1, initialSt:newSt, maskLen:mask_len, frmaskLen: mask_len_fr, enmaskLen: mask_len_en})
    losses.append(losses_new)
    if i%1000 == 0:
        print (sum(losses)/len(losses))


# In[ ]:


#test
nround = np.shape(en_test_token)[0]//bSz
right = 0
nw = 0
newSt1 = sess.run(initialSt1)
newSt = sess.run(initialSt)
for i in range(nround):
    fr_input = fr_test_token[i*bSz:(i+1)*bSz,0:wSz]
    en_input = en_test_token[i*bSz:(i+1)*bSz,0:wSz]
    en_ans = en_test_token[i*bSz:(i+1)*bSz,1:wSz+1]

    mask_len_fr = sequenceLen(fr_input, fr_dict, bSz)
    mask_len_en = sequenceLen(en_input, en_dict, bSz)
    mask_len = sequenceLen(en_ans, en_dict, bSz)

    pred, newSt1, newSt = sess.run([result, encSt, decSt], feed_dict = {encIn:fr_input, decIn:en_input, ans:en_ans, keepPrb:1, initialSt1:newSt1, initialSt:newSt, maskLen: mask_len, frmaskLen:mask_len_fr, enmaskLen:mask_len_en})
    #pred = np.reshape(pred, [-1,1])
    #en_ans = np.reshape(en_ans, [-1,1])
    for row in range(20):
        l = np.where(en_ans[row,:] == en_dict["<stop>"])[0]
        if len(l) == 0:
            l = 12
        else:
            l = l[0] + 1
        right += np.sum(np.equal(pred[row,range(l)],en_ans[row,range(l)]))/float(l)
	#nw += l
acc = right*100./(nround*bSz)

print "the accuracy is", acc


# In[ ]:





# In[ ]:




