
# coding: utf-8


import tensorflow as tf
import numpy as np
import time
import sys

train_path = sys.argv[1]
dev_path = sys.argv[2]
start = time.time()
#tokenization
train_data = open(train_path,"r").readlines()
train_tokens = []
for sentence in train_data:
    train_tokens.extend(sentence.split())


k = 0
token_dict = dict()
train_vocab = []
for word in train_tokens:
    if word in token_dict:
        train_vocab.append(token_dict[word])
    else:
        token_dict[word] = k
        k += 1
        train_vocab.append(token_dict[word])

train_input = train_vocab[0:len(train_vocab)-2]
train_input2 = train_vocab[1:len(train_vocab)-1]
train_ans = train_vocab[2:len(train_vocab)]


#define parameters
vocabSz = max(train_vocab) + 1
batchSz = 20
embedSz = 100
learning_rate = 1e-4


#placeholders
inpt = tf.placeholder(tf.int32, shape = [batchSz])
inpt2 = tf.placeholder(tf.int32, shape = [batchSz])
ans = tf.placeholder(tf.int32, shape = [batchSz])

#embedding
E = tf.Variable(tf.truncated_normal([vocabSz,embedSz], stddev = 0.1))

embed = tf.nn.embedding_lookup(E,inpt)
embed2 = tf.nn.embedding_lookup(E,inpt2)
both = tf.concat(axis=1,values=[embed,embed2])

#feed-forward layer
V = tf.Variable(tf.truncated_normal([embedSz*2,200], stddev = 0.1))
bV = tf.Variable(tf.truncated_normal([200], stddev = 0.1))
logit1 = tf.nn.relu(tf.matmul(both, V) + bV)
W = tf.Variable(tf.truncated_normal([200,vocabSz], stddev = 0.1))
b = tf.Variable(tf.truncated_normal([vocabSz], stddev = 0.1))
logit = tf.matmul(logit1,W) + b

probs = tf.nn.softmax(logits=logit, dim = -1)
xEnt = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logit, labels = ans)
loss = tf.reduce_sum(xEnt)

#train
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#perplecity
perplx = tf.exp(loss/batchSz)

sess= tf.Session()
sess.run(tf.global_variables_initializer())


#train NN
n = 0
losses = []
while n < (len(train_input) - 20):
    inpts = train_input[n:(n+batchSz)]
    inpts2 = train_input2[n:(n+batchSz)]
    anss = train_ans[n:(n+batchSz)]
    sess.run(train, feed_dict = {inpt: inpts, inpt2: inpts2, ans: anss})
    losses.append(sess.run(loss, feed_dict = {inpt: inpts, inpt2: inpts2, ans: anss}))
    n += 20

perlexity_train = np.exp(sum(losses)/(len(losses)*batchSz))
print "The Perlexity of Training Data is: %r" %(perlexity_train)

#tokenization dev data
dev = open(dev_path,"r").readlines()
dev_tokens = []
for sentence in dev:
    dev_tokens.extend(sentence.split())

dev_vocab = []
for word in dev_tokens:
    if word not in token_dict:
        dev_vocab.append(token_dict["*UNK*"])  
    else:
    	dev_vocab.append(token_dict[word])

dev_input = dev_vocab[0:len(dev_vocab)-2]
dev_input2 = dev_vocab[1:len(dev_vocab)-1]
dev_ans = dev_vocab[2:len(dev_vocab)]


# In[38]:


#test NN
n = 0
perplexity = []
while n < (len(dev_input) - 20):
    inpts = dev_input[n:(n+batchSz)]
    inpts2 = dev_input2[n:(n+batchSz)]
    anss = dev_ans[n:(n+batchSz)]
    perplexity.append(sess.run(loss, feed_dict = {inpt: inpts, inpt2: inpts2, ans: anss}))
    n += 20
    
perlexity = np.exp(sum(perplexity)/(len(perplexity)*batchSz))
print "The Perplexity of Development Data is: %r" %(perlexity)


# In[39]:


sess.close()

end = time.time() - start
print "The running time is: %r" %(end)


