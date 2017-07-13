# Oscar Saharoy 2017

import random
import numpy as np

class MatrixNet(object):

	''' MatrixNet object is an artificial neural network which will try to optimize itself to map the elements of self.input to those of self.output.
		It uses numpy arrays for parallel calculation and optimization. It has a simple multilayer perceptron architecture, so is of limited practical use
		other than as a learning tool. '''

	def __init__(self):

		# Defining activation function which allows network to behave in a more complex, non-linear fashion.
		# Derivative of activation function is used later for backpropagation calculation.

		self.activ  = lambda x: 1/(1+np.exp(-x))
		self.deriv  = lambda x: x*(1-x)

		# self.struct stores the structure of the network; the number of nodes in each layer.

		self.struct = [1,2,2]

		# input and target vectors stored in a numpy array. These can be changed to any desired values.

		self.input  = np.array([ [1.0],     [2.0]     ])
		self.target = np.array([ [0.3,0.2], [0.6,0.4] ])

		# self.net stores the value associated with each node in the network.
		# self.errors stores the error associated with each node, used to update weights in backpropagation.
		# self.deltas stores another arithemetic value to calculate the error on non-output nodes.

		self.net    = []
		self.errors = []
		self.deltas = []

		for i,layer in enumerate(self.struct):

			# Initializing lists so that we can assign into them later without IndexErrors.

			self.net     += [np.zeros((layer))]
			self.errors  += [np.zeros((layer))]
			self.deltas  += [np.zeros((layer))]
		
		# self.synaps stores each link, or synapse, between the nodes in the network.

		self.synaps = []
		
		for i,_ in enumerate(self.struct[:-1]):

			# Initializing weights to random values.
			# self.synaps contains one less array than self.net, which contains the values associated with each node.
			# This is beacuse the weights connect one layer of nodes to the next, so fall 'between' the layers.
			# Therefore, each array of weights contains as many rows as there are nodes in the previous layer,
			# and as many columns as there are nodes in the next layer.

			rows          = self.struct[i]
			cols          = self.struct[i+1]
			self.synaps  += [np.random.random((rows,cols))]

	def forwards(self):

		# Forward sweep of the network.

		# self.randex is a random index, relating to a random pair of input and target vectors.

		self.randex  = random.randint(0,len(self.target)-1)

		# Setting the first layer of nodes - the input nodes - to be equal to the input vector.

		self.net[0]  = self.input[self.randex]

		for i,_ in enumerate(self.net[:-1]):

			# For each layer, the value of the nodes in the next layer is equal to the matrix product of the
			# current layer and the weight matrix between the current layer and the next. The activation function
			# is applied to improve training and remove linearity from the network.

			layer         = self.net[i]
			synaps        = self.synaps[i]
			self.net[i+1] = self.activ(np.dot(layer,synaps))

		# The last layer of self.net is the output layer. The target vector as well as the output layer vector
		# are printed to compare them.

		print self.target[self.randex], self.net[-1]

	def backprop(self):

		# To reduce the difference between the actual output of the network and the target, we perform backpropagation.
		# This is an algorithm which calculates the derivative of the 'error' of the network with respect to each weight.
		# The weights are then updated to reduce the error, moving the output of the network closer to the target.

		# The network is iterated through in reverse, as errors are calulated at the output layer and then propagated
		# backwards through the network. [::-1] slices the list in such a way as to reverse it.

		for i in range(len(self.net))[::-1]:

			layer = self.net[i]

			# The calculation of errors is different depending on the layer.

			if layer is self.net[0]:

				# There is no layer of nodes before the input layer of nodes, so we skip this layer.

				continue
			
			elif layer is self.net[-1]:

				# For the output layer, the derivative of total error with respect to node output is the target
				# value minus the actual value.

				self.errors[i]    = self.target[self.randex] - self.net[i]

				# The derivative of total error with respect to node input is the error matrix multiplied by the
				# value of each node after being transformed bty self.deriv, the derivative of the activation function.

				self.deltas[i]    = self.errors[i] * self.deriv(self.net[i])

				# Finally, the product of this matrix with that of the values of the nodes in the layer before is taken
				# to calculate the derivative of total error with respect to each of the weights' values.
				# This value is added on to descend the gradient and reduce total error.

				self.synaps[i-1] += np.dot( np.array([self.net[i-1]]).T, np.array([self.deltas[i]]) )

			else:

				# The error on each node in the hidden layers is equal to the product of delta matrix from the layer above
				# and the weight matrix connecting the two layers.

				self.errors[i]    = np.dot(self.deltas[i+1],self.synaps[i].T)

				# The rest of the arithmatic is identical to that of the output layer.

				self.deltas[i]    = self.errors[i] * self.deriv(self.net[i])

				self.synaps[i-1] += np.dot( np.array([self.net[i-1]]).T, np.array([self.deltas[i]]) )


if __name__ == '__main__':

	net = MatrixNet()

	for _ in xrange(10000):

		net.forwards()
		net.backprop()