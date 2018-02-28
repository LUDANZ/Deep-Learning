**HW1: MNIST Neural Network From Scratch**
In this assignment, you will be coding a single-layer neural network to classify handwritten digits from scratch, using only NumPy. You will not be using TensorFlow, or any other deep learning framework, for this assignment.

**Network Architecture**
In this assignment, you will be building a single layer neural network from scratch with the following requirements:

Your network should take 784 values as input (representing the 28x28 image) and output the probabilities that the image belongs to each of the 10 class labels (one class for each digit from 0-9).
Your network should have a total of 7850 parameters. These are the weights wi,j and the biases bj where 0≤i≤783 and 0≤j≤9. All parameters should be initialized to 0.
Your network should be trained using the cross-entropy loss function. For a given training example, the error E=−log(pc) where pc is the probability of the correct answer in the example.
You should train your network on 10,000 training examples with a learning rate of λ=0.5. Note: The training dataset actually contains 60,000 images and while your network would likely perform better if trained on all the data, we are only requiring that 10,000 images be used so that the training time is reduced.
You should be coding the stochastic gradient descent (SGD) algorithm as discussed in class with a batch size of 1.
**Data**
You will be using the MNIST dataset to train and test your network. The dataset can be found here: http://yann.lecun.com/exdb/mnist/ . There are four files to download: two for training and two for testing.

The training data contains 60,000 examples broken into two files: one file contains the image pixel data and the other contains the class labels.

You should train your network using only the training data and then test your network's accuracy on the testing data. Your program should print its accuracy over the test dataset upon completion.

**Reading in the Data**
The MNIST data files are gzipped. You can use the gzip library to read these files from Python.

To open a gzipped file from Python you can use the following code:

import gzip
with open('file.gz', 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
  # bytestream contains the data of the fileobj
  # You can use bystream.read(n) to read n bytes from the file.
You might find the function numpy.frombuffer (https://docs.scipy.org/doc/numpy/reference/generated/numpy.frombuffer.html) helpful to convert from a buffer of bytes to a NumPy array.

Note: You should normalize the pixel values so that they range from 0 to 1 (This can easily be done by dividing each pixel value by 255) to avoid any numerical overflow issues.

**Data format**
The testing and training data are in the following format:

train-images-idx3-ubyte.gz: 16 byte header (which you can ignore) followed by 60,000 training images. A training example consists of 784 single-byte integers (from 0-255) which represent pixel intensities.

train-labels-idx1-ubyte.gz: 8 byte header (which you can ignore) followed by 60,000 training labels. A training label consists of single-byte integers from 0-9 representing the class label.

t10x-images-idx3-ubyte.gz: 16 byte header (which you can ignore) followed by 10,000 testing images.

t10x-labels-idx1-ubyte.gz: 8 byte header (which you can ignore) followed by 10,000 testing labels.

Note: You can use the data type np.uint8 for single-byte (or 8-bit) integers.

