# Oscar Saharoy 2018

import numpy

''' How to use this library:
	
	- import MatrixNet
	- create Network object eg: net = MatrixNet.Network(structure,inputset,targetset, print_cost=False)
	- call net.train(iterations) 
	- you can pass data though the network once it is trained like: output = net.forward(input_data)
	  Ensure input data is a numpy matrix of the correct length.
	- you can also add or chang data to the input or target sets by assinging to net.inputset or net.targetset eg:

		net.inputset = new_input_data
		net.targetset.append(new_target)
		

	structure is a List containing number of nodes in each layer: number of input 
	nodes must match length of input vectors and number of output nodes 
	must match length of target vectors.

Example structure = [2,3,2] 

	inputset and targetset are training data as 2 lists of numpy matrices.

	Example inputset and targetset format:

inputset  = [numpy.matrix([[  1.0,  2.0]]),
             numpy.matrix([[ -1.0,  4.0]]),
             numpy.matrix([[  0.3,  2.1]])]

targetset = [numpy.matrix([[  0.2,  0.4]]),
 	         numpy.matrix([[  1.0,  0.4]]),
 	         numpy.matrix([[  0.5,  1.0]])]

	Note: maximum network output on any node is 1.0 as sigmoid function is used. 

	print_cost specifies whether the cost at the end of each iteration over the
	training data should be printed to the console, and is a boolean value. '''


class Network(object):

	def __init__(self, structure, inputset, targetset, print_cost=False):

		self.structure  = structure
		self.inputset   = inputset
		self.targetset  = targetset
		self.print_cost = print_cost
 
		self.layers     = len(structure)  # Number of layers
 
		self.activate   = lambda x: 1/(1+numpy.exp(-x))  # Sigmoid activation function
		self.derivativ  = lambda x: x*(1-x)  # Derivative of activation function

		# Following lists initialised to copys of structure to ensure they have the correct number of elements.
		
		self.values    	= structure[:]  # List to store value of each node in network
		self.deltas    	= structure[:]  # List to store certain partial derivatives to be used in backpropogation
	
		self.weights   	= []  # List to store weight matrices
	
		# Creating inital weight matrices
		
		for i, n1 in enumerate(self.structure[:-1]):
		
			n2 = self.structure[i+1]
		
			weightrix = numpy.random.rand(n1,n2) * 0.2 - 0.1
		
			# Produces a matrix of random numbers between -0.1 and 0.1 with as many rows
			# as nodes in the previous layer and as many columns as nodes in the next layer
		
			self.weights += [weightrix]
	
	
	def forward(self,inp):

		# Forward sweep to bring input to output.

		values  = self.values
		weights = self.weights
	
		for layer in range(self.layers):
	
			if layer == 0:
	
				values[layer] = inp
	
				# For input layer, values of the nodes are equal to the input vector 
	
			else:
	
				values[layer] = self.activate(values[layer-1] * weights[layer-1])
	
				# For hidden and output layers, values of nodes are equal to
				# the weighted sum of the values of nodes in the previous layer.
				# This calculation is simplified by using a matrix multiplication.
				# Finally the activation function is applied.
	
		return values
	
	
	def backprop(self,tar):

		deltas    = self.deltas
		values    = self.values
		weights   = self.weights
	  
		arr       = numpy.array
		mat       = numpy.matrix

		if self.print_cost:

			# Calcuates error using squared error function if self.print_cost is True.

			out   = self.values[-1]
			error = sum([ (tar[0,x] - out[0,x])**2 for x in range(out.shape[0]) ])

	
		for layer in range(self.layers)[::-1]:
	
			if layer == 0:

				# Exit the loop when input layer is reached
	
				break
	

			# Calculates derivative of cost function with respect to weights in previous layer
			# using chain rule, and subtracts this differential from the weight matix to
			# decend the gradient of the cost landscape.
	
			elif layer == self.layers-1:
	
				deltas[layer] = arr(values[layer]-tar) * self.derivativ(arr(values[layer]))
	
			else:
	
				deltas[layer] = arr(deltas[layer+1] * weights[layer].T) * self.derivativ(arr(values[layer]))
	
	
			deltas[layer]     = mat(deltas[layer])
	
			weights[layer-1] -= (deltas[layer].T * values[layer-1]).T


		if self.print_cost:

			return error
	

	def train(self,iterate):

		# Creates variable to store error if print_cost is True.

		if self.print_cost:
			error = 0
		
		# Loop to call forward() and backprop() for each input and target vector pair,
		# repeating as many times as value of iterate.
	
		for i in range(iterate):
		
			ind    = i % len(self.inputset)
		
			inp    = self.inputset[ind]
			tar    = self.targetset[ind]
		
			values = self.forward(inp)

			e      = self.backprop(tar)
			
			if self.print_cost:

				# Add e to total error
				error += e

				# If end of training data is reached, print error and reset error to 0.
	
				if inp is self.inputset[-1]:
	
					print 'Cost',error
					error = 0
