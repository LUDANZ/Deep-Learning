
# coding: utf-8

# In[1]:


import tensorflow as tf
import gym
import numpy as np


# In[2]:


#call cart pole
env = gym.make('CartPole-v1')


# In[3]:


#parameters
hSz = 8
fSz = 2
learning_rate = 0.001


# In[4]:


#placeholder
stateIn = tf.placeholder(tf.float32, [None,4])
actionIn = tf.placeholder(tf.int32, [None])
rewardIn = tf.placeholder(tf.float32, [None])
TrewardIn = tf.placeholder(tf.float32, [None])


# In[5]:


#NN
W = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[4,hSz], stddev = 0.1))
bW = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[hSz], stddev = 0.1))
logit1 = tf.nn.relu(tf.matmul(stateIn, W) + bW) 
O = tf.Variable(tf.random_uniform(dtype=tf.float32, shape=[hSz, fSz]))
bO = tf.Variable(tf.random_uniform(dtype=tf.float32, shape=[fSz]))
logit = tf.matmul(logit1,O) + bO
final = tf.nn.softmax(logit, 1)
O1 = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[hSz, 1], stddev = 0.1))
bO1 = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[1], stddev = 0.1))
vOut = tf.matmul(logit1, O1) + bO1
vLoss = tf.reduce_mean(tf.square(TrewardIn - vOut))

indices = tf.range(0, tf.shape(final)[0]) * 2 + actionIn
actProb = tf.gather(tf.reshape(final,[-1]), indices)
loss = -tf.reduce_sum(tf.log(actProb) * rewardIn)
loss = loss + vLoss
train =  tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[6]:


#train
gamma = 0.99
R = []
for trial in range(3):
    totRewards = []
    for epi in range(1000):
        s = env.reset()
        Trewards = []
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
            Trewards.append(rwd)
            rewards.append((gamma**j)*rwd)
            acts.append(act)
            if j%50 == 0 and j != 0 and not dn:
                DisRewards = []
                for k in range(j+1):
                    DisRewards.append(sum(rewards[k:])/(gamma**k))
                sess.run(train, feed_dict={stateIn:states, actionIn:acts, rewardIn: DisRewards, TrewardIn: Trewards})
            if dn:
            #compute DisRewards and totRewards
                DisRewards = []
                for k in range(j+1): 
                    DisRewards.append(sum(rewards[k:])/(gamma**k))
                sess.run(train, feed_dict={stateIn:states, actionIn:acts, rewardIn: DisRewards, TrewardIn: Trewards})
                totRewards.append(j)
                break
            states.append(s)
    R.append(np.mean(totRewards[-100:]))

print "the mean reward collected for the last 100 episodes is", np.mean(R)


# In[ ]:




