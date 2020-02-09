import numpy as np
import random
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# import tensorflow as tf
# from tensorflow.contrib.distributions import Normal
# from keras.optimizers import Adam
# import keras as K
import pickle
# from collections import deque

# from sklearn.pipeline import FeatureUnion
# from sklearn.preprocessing import StandardScaler
# from sklearn.kernel_approximation import RBFSampler

			
class SGDRegressor:
	'''
	Class representing a single neuron
	'''
	def __init__(self, D, lr=0.1):
		self.w = np.random.randn(D) / np.sqrt(D)
		self.lr = lr

	def partial_fit(self, X, Y):
		self.w += self.lr*(Y - X.dot(self.w)).dot(X)

	def predict(self, X):
		return X.dot(self.w)

	def load(self, path):
		self.w = pickle.load(open(path,'rb'))

	def save(self,path):
		pickle.dump(self.w, open(path,'wb'))


class action_model:
	'''
	This model expects an input of type [position,speed]
	with position.shape = speed.shape
	'''
	def __init__(self,
				input_shape,
				alpha,
				gamma,
				lambda_):

		# if input_shape % 2 != 0:
		# 	print("action model input_shape parameter shouldn't be odd. You used:", input_shape)
		# 	quit()
		
		self.input_shape = input_shape
		self.alpha = alpha
		self.gamma = gamma
		self.lambda_ = lambda_
		self.traces = np.array([ np.zeros(input_shape), np.zeros(input_shape)])
		
		self.mean_layer = SGDRegressor(input_shape)
		self.stdv_layer = SGDRegressor(input_shape)
		
	def reset(self):
		self.traces = np.array([ np.zeros(input_shape), np.zeros(input_shape)])
	
	def predict(self, X, return_ms=False):
		mean = self.mean_layer.predict(X)
		stdv = np.exp(self.stdv_layer.predict(X))
		if return_ms:
			return np.random.normal(loc=mean, scale=stdv), mean, stdv
		else:
			return np.random.normal(loc=mean, scale=stdv)
	
	
	# def fit(self, X, delta):
	def fit(self, X, params_dict):
		X = X[0]
		action, mean, stdv = self.predict(X, True)
		dif = action-mean
		s2 = stdv*stdv
		delta = params_dict['delta']

		compat_p = dif * X / s2
		compat_s = (dif*dif/s2 - 1.) * X
				
		# 0 -> mean
		# 1 -> stdv
		self.traces[0] = self.traces[0] * self.gamma * self.lambda_
		self.traces[0] = self.traces[0] + compat_p
		self.traces[1] = self.traces[1] * self.gamma * self.lambda_
		self.traces[1] = self.traces[1] + compat_s
		
		self.mean_layer.w += self.alpha * delta * self.traces[0]
		self.stdv_layer.w += self.alpha * delta * self.traces[1]
	


class value_model:
	'''
	This model expects an input of type [position,speed]
	with position.shape = speed.shape
	'''
	def __init__(self,
				input_shape,
				alpha,
				gamma,
				lambda_):

		# if input_shape % 2 != 0:
		# 	print("action model input_shape parameter shouldn't be odd. You used:", input_shape)
		# 	quit()
		
		self.input_shape = input_shape
		self.alpha = alpha
		self.gamma = gamma
		self.lambda_ = lambda_
		self.traces = np.zeros(input_shape)
		
		self.only_layer = SGDRegressor(input_shape)
		
	def reset(self):
		self.traces = np.zeros(input_shape)
		
	def predict(self, X):
		return self.only_layer.predict(X)
	
	
	def fit(self, X, params_dict):
		X = X[0]
		delta = params_dict['delta']
		self.traces = self.traces * self.gamma * self.lambda_ + X
		self.only_layer.w += self.alpha * delta * self.traces

