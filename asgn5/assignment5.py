
# coding: utf-8



#load data
import tensorflow as tf
import numpy as np
import sys

train_path = sys.argv[1]
dev_path = sys.argv[2]

#tokenization
train_data = open(train_path,"r").readlines()
train_words = []
for s in train_data:
    train_words.extend(s.split())




word_dict = {w:id for id, w in enumerate(set(train_words))}
word_token = []
k = 0
for word in train_words:
    word_token.append(word_dict[word])



train_input = word_token[0:len(word_token)-1]
label_input = word_token[1:len(word_token)]




batchSz = 50
windowSz = 20
embedSz = 800
vocabSz = max(word_token) + 1
stateSz = 500




#placeholder
batch_input = tf.placeholder(dtype=tf.int32, shape=[batchSz, windowSz])
batch_label = tf.placeholder(dtype=tf.int32, shape=[batchSz, windowSz])

#embedding
embedding = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[vocabSz, embedSz],stddev = 0.1))
batch_in = tf.nn.embedding_lookup(embedding, batch_input)

#LSTM
lstm = tf.contrib.rnn.BasicLSTMCell(stateSz)
initialState = lstm.zero_state(batchSz, tf.float32)
output, nextState = tf.nn.dynamic_rnn(lstm, batch_in, initial_state = initialState)
output = tf.reshape(output,[-1,stateSz])

#prob
W = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[stateSz, vocabSz],stddev = 0.1))
b = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[vocabSz], stddev = 0.1))

#logit
logit = tf.matmul(output,W) + b
logit = tf.reshape(logit, [batchSz,windowSz,vocabSz])
#loss
weights = tf.ones(dtype=tf.float32, shape = [batchSz,windowSz])
loss = tf.contrib.seq2seq.sequence_loss(logits = logit, targets = batch_label, weights = weights)
train = tf.train.AdamOptimizer(1e-3).minimize(loss)

sess= tf.Session()
sess.run(tf.global_variables_initializer())





#train
def turn_batch(x,batchsz,windowsz):
    No_batch = len(x)//(batchsz*windowsz)
    x = np.array(x[0:batchSz*windowSz*No_batch])
    batches = np.reshape(x,[batchSz,-1])
    
    return batches

def new_batch(x,i):
    batch = x[:,i*windowSz:(i+1)*windowSz]
    
    return batch



#initial state
losses = []
train_batches = turn_batch(train_input,batchSz,windowSz)
label_batches = turn_batch(label_input,batchSz,windowSz)
batch_inputs = new_batch(train_batches,0)
batch_labels = new_batch(label_batches,0)
state_new = sess.run(initialState)

sess.run(train, feed_dict = {batch_input:batch_inputs, batch_label:batch_labels, initialState:state_new})
losses_new, state_new = sess.run([loss, nextState], feed_dict = {batch_input:batch_inputs, batch_label:batch_labels, initialState:state_new})
losses.append(losses_new)

i = 1
while (i+1)*batchSz*windowSz < len(train_input):
    batch_inputs = new_batch(train_batches,i)
    batch_labels = new_batch(label_batches,i)
    sess.run(train, feed_dict = {batch_input:batch_inputs, batch_label:batch_labels, initialState:state_new})
    losses_new, state_new = sess.run([loss, nextState], feed_dict = {batch_input:batch_inputs, batch_label:batch_labels, initialState:state_new})

    losses.append(losses_new)
    if i%100 == 0:
        print "loss after %r iteration is: %s" %(i,sum(losses)*batchSz*windowSz/i)
    
    i += 1

print "perplxity of training set is %r" %(np.exp(sum(losses)/len(losses)))





#dev
dev_data = open(dev_path,"r").readlines()
dev_words = []
for s in dev_data:
    dev_words.extend(s.split())
    
dev_token = []
for word in dev_words:
    if word not in word_dict:
        dev_token.append(word_dict["*UNK*"])
    else:
        dev_token.append(word_dict[word])
        
dev_input = dev_token[0:len(dev_token)-1]
dev_label = dev_token[1:len(dev_token)]



i = 0
losses_dev = []
dev_batches = turn_batch(dev_input,batchSz,windowSz)
devlabel_batches = turn_batch(dev_label,batchSz,windowSz)
while (i+1)*batchSz*windowSz < len(dev_input):
    dev_batch_inputs = new_batch(dev_batches,i)
    dev_batch_labels = new_batch(devlabel_batches,i)
    losses_dev_new = sess.run(loss, feed_dict = {batch_input:dev_batch_inputs, batch_label:dev_batch_labels, initialState:state_new})
    losses_dev.append(losses_dev_new)
    i += 1

print "perplxity of development set is %r" %(np.exp(sum(losses_dev)/len(losses_dev)))


sess.close()




