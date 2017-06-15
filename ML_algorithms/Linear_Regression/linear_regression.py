#Implementing Linear regression in tensor flow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# np.arange() Return evenly spaced values within a given interval.
# self genereated sinusoidal data
X_data = np.arange(0, 100, 0.1)
Y_data = X_data + 15 * np.sin(X_data/10)

# X_data_test = np.arange(0, 100, 0.1)
# Y_data_test = X_data_test + 15 * np.sin(X_data_test/10)

# no_test_samples = X_data_test.shape[0]
# X_data_test = np.reshape(X_data_test, (no_test_samples, 1))
# Y_data_set = np.reshape(Y_data_test, (no_test_samples, 1))

#plt.scatter(X_data, Y_data)
#plt.show()

no_samples = 1000
X_data = np.reshape(X_data, (no_samples, 1))
Y_data = np.reshape(Y_data, (no_samples, 1))

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

batch_size = 250
learning_rate = 0.001
no_epochs = 4001
cost_history = np.empty(shape=[1], dtype=float)

# define placeholders for training data
X = tf.placeholder(tf.float32, shape=(batch_size, 1))
Y = tf.placeholder(tf.float32, shape=(batch_size, 1))

# testing time - you want to find predicted value on all data
# and not on a single batch size
X_t = tf.placeholder(tf.float32, shape=(None, 1))

# define variables which are the learning parameters of
# our linear regression model

# tf.truncated_normal selects random numbers from a normal distribution
# whose mean is close to 0 and values are close to 0.
# Ex. -0.1 to 0.1. It's called truncated because your cutting off the
# tails from a normal distribution.

# tf.random_normal selects random numbers from a normal distribution
# whose mean is close to 0; however the values can be a bit further
# apart. Ex. -2 to 2

# In practice (Machine Learning) you usually want your weights to be
# close to 0.


W = tf.Variable(tf.random_normal((1,1)), name = 'weights')
b = tf.Variable(tf.random_normal((1,)), name = 'bias')

Y_pred = tf.matmul(X, W) + b

# RMSE Loss function
loss = tf.reduce_mean((Y_pred - Y)**2)

# Using Adam optimizer - a variant of GD as optimizer
optimizer_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print 'Training starts'
        for epoch in range(no_epochs):
                # select batches randomly
                indices = np.random.choice(X_train.shape[0], batch_size)
                X_batch, Y_batch = X_train[indices], Y_train[indices]

                # running gradient descent
                _, cost_val = sess.run([optimizer_opt, loss], feed_dict = {X: X_batch, Y: Y_batch})
                cost_history= np.append(cost_history, cost_val)
                if epoch % 50 == 0:
                        print 'Epoch: ', epoch, ', Loss: ', cost_val

        # Plot the predicted values
        test_pred = tf.matmul(X_t, W) + b
        # Y_test here is the corresponding predicted value for given sinusoidal data
        Y_t = sess.run(test_pred, feed_dict={X_t: X_test})
        '''
        print'The original and predicted value are:'
        for x,y in zip(Y_data, Y_test):
                print x, ': ',y
        '''
        #print 'test time MSE is'
        mse = tf.reduce_mean((Y_t - Y_test)**2)

        print("Test MSE: %.4f" % sess.run(mse))
        # print 'Cost history is ', cost_history

        print 'Actual Y value is\n', Y_test
        print 'Predicted Y value is\n', Y_t

# The predicted MSE comes out to be ~86.32                                                                 