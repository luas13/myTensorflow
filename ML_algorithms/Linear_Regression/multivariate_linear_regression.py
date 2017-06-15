'''
Using Boston Dataset: https://archive.ics.uci.edu/ml/datasets/Housing

Number of Instances: 506
Number of Attributes: 13 continuous attributes (including "class"
                         attribute "MEDV"), 1 binary-valued attribute.
Last column(14th) is the target cost

Attribute Information:

    1. CRIM      per capita crime rate by town
    2. ZN        proportion of residential land zoned for lots over
                 25,000 sq.ft.
    3. INDUS     proportion of non-retail business acres per town
    4. CHAS      Charles River dummy variable (= 1 if tract bounds
                 river; 0 otherwise)
    5. NOX       nitric oxides concentration (parts per 10 million)
    6. RM        average number of rooms per dwelling
    7. AGE       proportion of owner-occupied units built prior to 1940
    8. DIS       weighted distances to five Boston employment centres
    9. RAD       index of accessibility to radial highways
    10. TAX      full-value property-tax rate per $10,000
    11. PTRATIO  pupil-teacher ratio by town
    12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks
                 by town
    13. LSTAT    % lower status of the population
    14. MEDV     Median value of owner-occupied homes in $1000's

Multivariate Linear regression is implemented with 13 features
'''

import tensorflow as tf
import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

def normalize(dataset):
        mu = np.mean(dataset, axis=0)
        sd = np.std(dataset, axis=0)
        return (dataset - mu)/sd

# Read the data
boston_data = load_boston()
features = np.array(boston_data.data)
labels = np.array(boston_data.target)

# Number of samples and dimensions
no_samples = features.shape[0]
no_dim = features.shape[1]

# Normalize the features as a pre-processing step
normalised_features = normalize(features)

'''
Either use a bias variable to add later in tensorflow graph or append now

# Append a bias term to create final normalised features and labels
f_features = np.reshape(np.c_[normalised_features, np.ones(no_samples)], (no_samples, no_dim+1))
f_labels = np.reshape(labels, (no_samples, 1))
'''

f_features = np.reshape(normalised_features, (no_samples, no_dim))
f_labels = np.reshape(labels, (no_samples, 1))

# Randomly create train and test datasets
X_train, X_test, Y_train, Y_test = train_test_split(f_features, f_labels, test_size=0.20, random_state=42)


# Now the model and training part
batch_size = 250
learning_rate = 0.01
no_epochs = 4001
cost_history = np.empty(shape=[1], dtype=float)

# define placeholders for training data
X = tf.placeholder(tf.float32, shape=(batch_size, no_dim))
Y = tf.placeholder(tf.float32, shape=(batch_size, 1))

# testing time - you want to find predicted value on all data
# and not on a single batch size
X_t = tf.placeholder(tf.float32, shape=(None, no_dim))


# Define the learning parameters i.e. weights and bias
# W = tf.Variable(tf.random_normal((1,1)), name='weights')
# b = tf.Variable(tf.random_normal((1,)), name='bias')

W = tf.Variable(tf.ones([no_dim, 1]), name='weights')
b = tf.Variable(tf.ones([1]), name='bias')

Y_pred = tf.matmul(X, W) + b

# loss = tf.reduce_sum((Y_pred - Y)**2/batch_size)
loss = tf.reduce_mean((Y_pred - Y)**2)


optimizer_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(no_epochs):

                # select batches randomly
                indices = np.random.choice(X_train.shape[0], batch_size)
                X_batch, Y_batch = X_train[indices], Y_train[indices]

                # Running gradient descent
                _, cost_val = sess.run([optimizer_opt, loss], feed_dict={X: X_batch, Y: Y_batch})
                cost_history= np.append(cost_history, cost_val)

                if epoch%50 == 0:
                        print 'Epoch: ', epoch, ', Loss: ', cost_val


        # Testing phase
        test_pred = tf.matmul(X_t, W) + b
        # Y_test here is the corresponding predicted value for given sinusoidal data
        Y_t = sess.run(test_pred, feed_dict={X_t: X_test})

        # mse = tf.reduce_sum((Y_t - Y_test)**2 / Y_test.shape[0])
        mse = tf.reduce_mean((Y_t - Y_test)**2)

        print("MSE: %.4f" % sess.run(mse))
        print 'Cost history is ', cost_history
        print 'Actual Y value is\n', Y_test
        print 'Predicted Y value is\n', Y_t

# The predicted MSE comes out to be ~25.144

'''
fig, ax = plt.subplots()
ax.scatter(Y_t, Y_test)
ax.plot([Y_t.min(), Y_t.max()], [Y_t.min(), Y_t.max()], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
'''