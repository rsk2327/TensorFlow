import pandas as pd 
import tensorflow as tf 
import os
import numpy as np

train = np.array([[1,3],[2,4],[5,3],[3,5],[6,5],[2,8],[5,8],[3,9],[3,4],[2,4]])
test = np.array([[2,2],[4,4],[5,7],[7,4]])

trainLabel = np.array([11,16,19,21,27,28,34,33,18,16])
testLabel = np.array([10,20,31,26])

n_samples = train.shape[0] 
inputSize = train.shape[1] 

x = tf.placeholder(tf.float32,[None,inputSize])
y_ = tf.placeholder(tf.float32,[None])


def create_network(hiddenSize):
	'''
	hiddenSize - array with number of layers and their sizes

	'''

	numLayers = len(hiddenSize)

	W=[0 for i in range(numLayers+1)]
	b=[0 for i in range(numLayers+1)]
	layers=[0 for i in range(numLayers)]

	W[0] = tf.Variable(tf.random_normal([inputSize,hiddenSize[0]])) 
	b[0] = tf.Variable(tf.zeros([hiddenSize[0]])) 

	layers[0] = tf.add( tf.matmul(x,W[0]) , b[0] )
	layers[0] = tf.nn.sigmoid(layers[0])

	for i in range(1,numLayers):

		W[i] = tf.Variable(tf.random_normal([hiddenSize[i-1],hiddenSize[i]])) 
		b[i] = tf.Variable(tf.zeros([hiddenSize[i]])) 

		layers[i] = tf.add( tf.matmul(layers[i-1],W[i]) , b[i] )
		layers[i] = tf.nn.sigmoid(layers[i])

	W[numLayers] = tf.Variable(tf.random_normal([hiddenSize[numLayers-1], ])) 
	b[numLayers] = tf.Variable(np.random.randn())

	pred = tf.add( tf.reduce_sum(tf.multiply(layers[numLayers-1],W[numLayers]),1) , b[numLayers])

	return pred


pred = create_network([100,50,5])


cost = tf.reduce_sum(tf.pow(pred-y_, 2))/(2*n_samples)

learning_rate = 0.01

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

numIter = 15000

for i in range(numIter):
	c = sess.run(cost, feed_dict = {x:train, y_:trainLabel})
	preds = sess.run(pred, feed_dict = {x:train})
	
	if i%50==0:
		print("cost=", "{:.9f}".format(c))
	
	sess.run(optimizer, feed_dict = {x: train, y_: trainLabel} )

	
c = sess.run(cost, feed_dict = {x:train, y_:trainLabel})
preds = sess.run(pred, feed_dict = {x:test})
print("cost=", "{:.9f}".format(c), preds)

