# Oscar Saharoy 2017

import random
import numpy as np

class MatrixNet(object):

	def __init__(self):

		self.activ  = lambda x: 1/(1+np.exp(-x))
		self.deriv  = lambda x: x*(1-x)

		self.struct = [1,2,2]

		self.input  = np.array([ [1.0],     [2.0]     ])
		self.target = np.array([ [0.3,0.2], [0.6,0.4] ])

		self.net    = []
		self.errors = []
		self.deltas = []

		for i,layer in enumerate(self.struct):

			self.net     += [np.random.random((layer))]
			self.errors  += [np.random.random((layer))]
			self.deltas  += [np.random.random((layer))]

		self.synaps = []
		
		for i,_ in enumerate(self.struct[:-1]):

			layer         = self.struct[i]
			nextlayer     = self.struct[i+1]
			self.synaps  += [np.random.random((layer,nextlayer))]

	def forwards(self):

		self.randex = random.randint(0,len(self.target)-1)

		self.net[0] = self.input[self.randex]

		for i,_ in enumerate(self.net[:-1]):

			layer         = self.net[i]
			synaps        = self.synaps[i]
			self.net[i+1] = self.activ(np.dot(layer,synaps))

		print self.target[self.randex], self.net[-1]

	def backprop(self):

		for i in range(len(self.net))[::-1]:

			layer   = self.net[i]
			
			if layer is self.net[-1]:

				self.errors[i]    = self.target[self.randex] - self.net[i]
				self.deltas[i]    = self.errors[i] * self.deriv(self.net[i])

				self.synaps[i-1] += np.dot(np.array([self.net[i-1]]).T,np.array([self.deltas[i]]))

			elif layer is self.net[0]:

				pass

			else:

				self.errors[i]    = np.dot(self.deltas[i+1],self.synaps[i].T)
				self.deltas[i]    = self.errors[i] * self.deriv(self.net[i])
				
				self.synaps[i-1] += np.dot(np.array([self.net[i-1]]).T,np.array([self.deltas[i]]))

if __name__ == '__main__':

	net = MatrixNet()

	for _ in xrange(10000):

		net.forwards()
		net.backprop()