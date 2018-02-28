# Assignment 3: Convolutional Neural Networks

## Goal:

We will be building a Convolutional Neural Network (CNN) with two convolution & max pooling layers for the MNIST digit classification task.

## Steps:

Set up placeholders for inputs and outputs.
Initialize parameters for the model.
Weight variables should be initialized from a normal distribution (tf.truncated_normal) with a standard deviation of 0.1.
Implement the forward pass for the model:
Convolution Layer 1 (5 x 5 filter, 32 deep, 1 in channel) [tf.nn.conv2d]
ReLU Nonlinearlity [tf.nn.relu]
Max Pooling 1 (strides = [1, 2, 2, 1]) [tf.nn.max_pool]
Convolution Layer 2 (5 x 5 filter, 64 deep, 32 in channels)
ReLU Nonlinearity
Max Pooling 2 (strides = [1, 2, 2, 1])
Some number of feed-forward layers, into final softmax layer (as we had before). Pick the layer sizes and number of layers to maximize accuracy.
Calculate the loss for the model:
Calculate cross-entropy loss.
Set up the training step:
Use a learning rate of 1E-4
Use an Adam Optimizer [tf.train.AdamOptimizer]

## Notes:

Train for 2000 batches, with a batch size of 50. You must abide by this - we will not grade any submission that uses an alternate batch size, or number of batches.
