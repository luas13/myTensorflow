#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# notice we use the same model as linear regression, this is because there is a 
# baked in cost function which performs softmax and cross entropy
def model(X, w):
    return tf.matmul(X, w) 

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784]) # create symbolic variables
Y = tf.placeholder("float", [None, 10])

# like in linear regression, we need a shared variable weight matrix for logistic regression
w = init_weights([784, 10]) 
py_x = model(X, w)

hypothesis = tf.div(1.0, 1.0 + tf.exp(-py_x))

# Cost
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

#eta = tf.Variable(0.01)
eta = tf.Variable(0.13)
no_iterations = 400
batch_size = 500

optimizer = tf.train.GradientDescentOptimizer(eta)
train_op = optimizer.minimize(cost)

# Start all variables after execute nodes
# init = tf.global_variables_initializer()

predict_op = tf.argmax(py_x, 1)

'''
# compute mean cross entropy (softmax is applied internally)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) 
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
predict_op = tf.argmax(py_x, 1) # at predict time, evaluate the argmax of the logistic regression
'''

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for iteration in range(no_iterations + 1):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)

        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(train_op, feed_dict={X: batch_xs, Y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys})/total_batch

        # Display logs per eiteration step
        if iteration % 20 == 0:
            print "Iteration:", '%04d' % (iteration), "cost=", "{:.9f}".format(avg_cost)

        '''
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        if i%10 == 0:
            print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX})))
        '''

    '''
    # Test the model
    predictions = tf.equal(tf.argmax(py_x, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print "Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels})
    '''
    print 'Accuracy: ', np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX}))

    
