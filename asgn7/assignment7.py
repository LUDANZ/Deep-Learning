
# coding: utf-8

# In[1]:


import tensorflow as tf
import gym
import numpy as np


# In[2]:


#call cart pole
env = gym.make('CartPole-v0')


# In[3]:


#parameters
hSz = 8
fSz = 2
learning_rate = 0.005


# In[4]:


#placeholder
stateIn = tf.placeholder(tf.float32, [None,4])
actionIn = tf.placeholder(tf.int32, [None])
rewardIn = tf.placeholder(tf.float32, [None])


# In[5]:


#NN
W = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[4,hSz], stddev = 0.1))
bW = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[hSz], stddev = 0.1))
logit1 = tf.nn.relu(tf.matmul(stateIn, W) + bW) 
O = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[hSz, fSz], stddev = 0.1))
bO = tf.Variable(tf.truncated_normal(dtype=tf.float32, shape=[fSz], stddev = 0.1))
logit = tf.matmul(logit1,O) + bO
final = tf.nn.softmax(logit, 1)

indices = tf.range(0, tf.shape(final)[0]) * 2 + actionIn
actProb = tf.gather(tf.reshape(final,[-1]), indices)
loss = -tf.reduce_sum(tf.log(actProb) * rewardIn)
train =  tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[7]:


#train
gamma = 0.999
R = []
for trial in range(3):
    totRewards = []
    for epi in range(1000):
        s = env.reset()
        rewards = []
        states = []
        acts = []
        DisRewards = []
        states.append(s)
        for j in range(1000):
            #env.render()
            actProbs = sess.run(final, feed_dict={stateIn:[s]})
            if np.random.rand(1)[0]<=actProbs[0][0]: 
                act = 0
            else:
                act = 1
            s1,rwd,dn,_ = env.step(act)
            s = s1
            rewards.append((gamma**j)*rwd)
            acts.append(act)
            if dn:
            #compute DisRewards and totRewards
                for k in range(j+1):
                    DisRewards.append(sum(rewards[k:])/(gamma**k))
                sess.run(train, feed_dict={stateIn:states, actionIn:acts, rewardIn: DisRewards})
                totRewards.append(j)
                break
            states.append(s)
    R.append(np.mean(totRewards[-100:]))

print "the mean reward collected for the last 100 episodes is", np.mean(R)


# In[ ]:




