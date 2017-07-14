# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 00:43:44 2017

@author: Da Xu

This model is a RNN model with LSTM cells to predict S&P500 
"""

from __future__ import print_function
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.contrib import rnn

#read data from the historical data file of s&p500

fields = [1,2,3,4,5,6]

df = pd.read_csv('../historicalData/^GSPC-1.csv', header=None, skiprows =1, usecols=fields, nrows =100000)

df.loc[:,1] *= 0.001
df.loc[:,2] *= 0.001
df.loc[:,3] *= 0.001
df.loc[:,4] *= 0.001
df.loc[:,5] *= 0.001
df.loc[:,6] *= 0.00000000001


data = df.values

h, w = data.shape

n_steps = 50
n_echo = 5
batch_size = 40
epochs = 25



s=(h-n_echo-n_steps+1,50,6)
hist_s = np.zeros([h-n_echo-n_steps+1,50,6])
toPredict_s = np.zeros([h-n_echo-n_steps+1,6]) 

for i in range(0, h-n_echo-n_steps+1):
    hist_s[i][:][:] = data[i:(i+n_steps),:]
 
for i in range(0, h-n_echo-n_steps+1):
    toPredict_s[i][:][:] = data[i+n_steps+n_echo-1,:]
    
hist_prices = tf.constant(hist_s)    
toPredict = tf.constant(toPredict_s)
n_samples = h-n_echo-n_steps+1


'''
To predict 5 days later open, high, low, close, volume and adjust close, we use
50 day historical data as the input in the RNN model. Therefore the time steps
is 50.
'''

# Parameters
learning_rate = 0.001
n_test_samples = 1000
training_iters = n_samples - n_test_samples

display_step = 10

# Network Parameters
n_input = w # MNIST data input (img shape: 28*28)
n_hidden = 128 # hidden layer num of features
n_outputs = 6 # open high low close  volume adj close

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_outputs])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_outputs]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_outputs]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.squared_difference(pred, y))
cost_prediction = tf.reduce_mean(tf.squared_difference(pred[4], y[4]))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph


with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations  
    
    for i in range(epochs):
        step = 1
        print ("Epoch ", i)
        while step * batch_size < training_iters:
            batch_x = hist_prices[(step-1)*batch_size:(step*batch_size)][:][:]
            batch_y = toPredict[(step-1)*batch_size:(step*batch_size)][:]
        
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x.eval(), y: batch_y.eval()})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: batch_x.eval(), y: batch_y.eval()})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x.eval(), y: batch_y.eval()})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss))
            step += 1
        
    print("Optimization Finished!")

    # Calculate accuracy for test prices 
    test_hist_prices = hist_prices[(n_samples-n_test_samples):n_samples][:][:]
    test_toPredict = toPredict[(n_samples-n_test_samples):n_samples][:]
    testLoss = sess.run(cost_prediction, feed_dict={x: test_hist_prices.eval(), y: test_toPredict.eval()})
    print("Testing loss:", testLoss)
   
    print(test_toPredict.eval())
    ss = 0.0    
    for i in range(n_test_samples-n_echo):
        ss = ss+ (test_toPredict.eval()[i][4]-test_toPredict.eval()[i+n_echo][4])**2
    print("the variance of outputs ",  ss/(n_test_samples-n_echo))
    print("prediction of ", n_echo, "days")
