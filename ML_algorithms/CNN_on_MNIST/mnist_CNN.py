# This model gives us an accuracy of 99.18 % on MNIST dataset
# coding=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# tf.truncated_normal
# Random values with a normal distribution but eliminating those values whose
# magnitude is more than 2 times the standard deviation

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev= 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape= shape)
    return tf.Variable(initial)

"""
x = Input tensor of shape [batch, in_height, in_width, in_channels]
W = a filter/kernel tensor of shape
    [filter_height, filter_width, in_channels, out_channels]

strides = [batch, height, width, channels]
the convolution operates on a 2D window on the height, width dimensions.

strides = [1, 1, 1, 1] applies the filter to a patch at every offset,
strides = [1, 2, 2, 1] applies the filter to every other image patch
in each dimension, etc.
"""

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding= 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            padding='SAME')

"""
A ConvNet is made up of Layers. Every Layer has a simple API:
It transforms an input 3D volume to an output 3D volume with some
differentiable function that may or may not have parameters.
"""

def cnn():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    """
    values we will input during computation 

    The tensor x will be used to store the MNIST images as a vector of 
    784 floating point values (using None we indicate that the dimension
    can be any size; in our case it will be equal to the number of elements
    included in the learning process).
    """

    x = tf.placeholder("float", shape=[None, 784])
    y = tf.placeholder("float", shape=[None, 10])

    """
    Here we changed the input shape to a 4D tensor, the second and third dimension
    correspond to the width and the height of the image while the last dimension
    corresponding number of colour channels, 1 in this case.
    """

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    """
    In particular, unlike a regular Neural Network, the layers of a
    ConvNet have neurons arranged in 3 dimensions: width, height, depth.

    ********* 1st Layer: **********
    Convolution layer with max pooling
    [width, height, depth, output_size]

    output_size here means no of filters to use/ no of
    features to extract in this layer.
    Thus we are using 32 no of 5x5 filter with depth 1
	
    Each neuron of our hidden layer is connected with a small 5×5 region
    of the input layer.
	
    Shared matrix W and the bias b are usually called a kernel or filter 
    in the context of CNN’s. These filters are similar to those used image 
    processing programs for retouching images, which in our case are used 
    to find discriminating features.
	
    Neurons in each depth slice use same weights and bias.
	
    A weight matrix and a bias define a kernel. A kernel only detects one 
    certain relevant feature in the image so it is, therefore, recommended 
    to use several kernels, one for each characteristic we would like to detect.
	
    The first hidden layer is composed by several kernels. In our example, 
    we use 32 kernels, each one defined by a 5×5 weight matrix W and a bias b
    that is also shared between the neurons of the hidden layer.
    """

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    """
    Max POOL layer will perform a downsampling operation along the
    the spatial dimensions (width, height), resulting in volume
    such as [14x14x32] since we are using a window of size 2x2

    ********* 2nd layer: *********
    Convolution layer with max pooling
    [width, height, depth, output_size]

    output_size here means no of filters to use/ no of
    features to extract in this layer
    Thus we are using 64 no of 5x5 filter with depth 32
	
    In this case we have to pass 32 as the number of channels that we 
    need as that is the output size of the previous layer
    """

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    """
    Max POOL layer will perform a downsampling operation along the
    the spatial dimensions (width, height), resulting in volume
    such as [7x7x64] since we are using a window of size 2x2

    Applying 2×2 max pooling we are shrinking the image
    After 2 layers we moved from 28×28 image to 7×7
    For each point we have 64 features

    ********* 3rd layer: *********
    Fully connected layer
    [input_size, output_size]
	
    We will use a layer of 1024 neurons, allowing us to to process
    the entire image.
	
    First dimension of the tensor represents the 64 filters of size 7×7
    from the second convolution layer, while the second parameter is the
    amount of neurons in the layer and is free to be chosen by us (in our case 1024).
    
    More explanation about 7x7x64:
    Working from the start:
    The input, _X is of size [28x28x1] (ignoring the batch dimension). 
    A 28x28 greyscale image. The first convolutional layer uses PADDING=same, 
    so it outputs a 28x28 layer, which is then passed to a max_pool with k=2, 
    which reduces each dimension by a factor of two, resulting in a 14x14 spatial 
    layout. conv1 has 32 outputs -- so the full per-example tensor is now [14x14x32].

    This is repeated in conv2, which has 64 outputs, resulting in a [7x7x64].
    tl;dr: The image starts as 28x28, and each maxpool reduces it by a factor 
    of two in each dimension. 28/2/2 = 7
    
    """

    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])

    # flattening the output of previous layer into a vector.
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    """
    Add dropout
	
    Dropout reduces the risk of the model of overfitting the data. This can 
    happen when the hidden layers have large amount of neurons and thus can 
    lead to very expressive models; in this case it can happen that random 
    noise (or error) is modelled. This is known as overfitting, which is more 
    likely if the model has a lot of parameters compared to the dimension of 
    the input. The best is to avoid this situation, as overfitted models have 
    poor predictive performance.
    """

    # using a placeholder for keep_prob will allow us to turn off
    # dropout during the testing
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    """
    ********* 4th layer: *********
    Fully connected layer
    [input_size, output_size]
    """

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_hat = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    """
    Define the Cost
    cross entropy loss function
    """
    cross_entropy = -tf.reduce_sum(y*tf.log(y_hat))

    """
    Define the training algorithm
    Minimization of cross entropy with adaptive gradient
    Learning rate is 1e-4
    """
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    """
    Start a new Session
    """
    sess = tf.InteractiveSession()

    # Initialize all the variables
    sess.run(tf.initialize_all_variables())

    """
    Define the accuracy before training
    1 is axis here.
    tf.argmax Returns the index with the largest value across axes of a tensor.
	
    tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
    This instruction returns a list of Booleans. To determine which fractions
    of predictions are correct, we can cast the values to numeric variables 
    (floating point) 
	
    For example, [True, False, True, True] will turn into [1,0,1,1] and 
    the average will be 0.75 representing the percentage of accuracy.
    """
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    """
    Train the model with 20000 epochs
    """
    for n in range(20000):
        batch = mnist.train.next_batch(50)
        if n%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            print "step %d, training accuracy %g" % (n, train_accuracy)
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

    #Evaluation Accuracy

    print "test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y:mnist.test.labels, 
        keep_prob: 1.0})


if __name__=='__main__':
    cnn()
