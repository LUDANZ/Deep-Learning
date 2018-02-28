##Homework 2: MNIST Neural Network with Tensorflow

In this assignment, you will be improving the network from HW1 using TensorFlow.

First, you should rewrite the network from HW1 using TensorFlow. Then, to improve accuracy, you should add another hidden layer to the network before the softmax layer. Finally, you should experiment with the network's hyperparameters (batch size, learning rate & hidden layer size) to find those the give the best results.

##Assignment Details
Your network's accuracy should be greater than 92% on the test dataset.
Your network must now include an extra hidden layer. You can choose any size for the hidden layer as long as you reach the accuracy target.
You can initialize the model's parameters in any way you like as long as you reach the accuracy target.
You can choose any learning rate and batch size (although the batch size must be greater than 1) as long as the accuracy target is met.
You can train on as many of the 60,000 examples as you like (and even use multiple epochs) as long as your program runs in under 2 minutes on a department machine.

##Notes

Tensorflow provides a nice interface with interacting with the MNIST data (from tensorflow.examples.tutorials.mnist import input_data). You will want to use this to load the data, and you will want to use the provided next_batch function during training.
You can find the full Tensorflow Python API documentation here.
