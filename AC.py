import gym
import numpy as np
import random
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# import tensorflow as tf
# import keras as K
import matplotlib.pyplot as plt
import pickle
import os
from collections import deque
import sys

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from dqn_models import SGDRegressor, action_model, value_model

# class FeatureTransformer:
# 	#  def __init__(self, env, n_components=500):
# 	def __init__(self, observation_examples, n_components=500):
# 		# observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
# 		scaler = StandardScaler()
# 		scaler.fit(observation_examples)

# 		# Used to converte a state to a featurizes represenation.
# 		# We use RBF kernels with different variances to cover different parts of the space
# 		featurizer = FeatureUnion([
# 			("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
# 			("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
# 			("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
# 			("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
# 			])
# 		example_features = featurizer.fit_transform(scaler.transform(observation_examples))

# 		self.dimensions = example_features.shape[1]
# 		self.scaler = scaler
# 		self.featurizer = featurizer

# 	def transform(self, observations):
# 		# print(observations.shape)
# 		# quit()
# 		scaled = self.scaler.transform(observations)
# 		return self.featurizer.transform(scaled)

#################################################################################################

class FeatureTransformer:
	#  def __init__(self, env, n_components=500):
	def __init__(self, observation_examples, n_components=500):
		self.numTilings = 10
		self.minInput = [-1.2,-0.07]
		self.maxInput = [0.6,0.07]

		# tilesPerTiling is in deed #tiles/side of tiling
		# that is, #tiles/tiling = tilesPerTiling**2
		self.tilesPerTiling = 10
		self.t2 = self.tilesPerTiling*self.tilesPerTiling
		self.tileSize = [0.,0.]
		self.tilingOffset = [0.,0.]
		for i in range(2):
			self.tileSize[i] = (self.maxInput[i]-self.minInput[i]) / (self.tilesPerTiling -1)
			self.tilingOffset[i] = self.tileSize[i] / self.numTilings 

	def transform(self, observations):
		in1 = observations[0][0] - self.minInput[0]
		in2 = observations[0][1] - self.minInput[1]
		tileIndices = [-1]*self.numTilings
		for tiling in range(self.numTilings):
			x = int(in1 / self.tileSize[0])
			y = int(in2 / self.tileSize[1])
			index = (y * self.tilesPerTiling + x) + tiling * self.t2
			tileIndices[tiling] = index
			in1+=self.tilingOffset[0]
			in2+=self.tilingOffset[1]
		
		sparsevec = [0 for i in range(self.tilesPerTiling * self.t2 + 1)]
		for idx in tileIndices:
			sparsevec[idx] = 1
		sparsevec[-1] = 1
		return np.array([sparsevec])

#################################################################################################

class DQN:
	def __init__(self, env, featurizer, n_features):
		self.memory = deque(maxlen=2000)
		
		self.input_shape = n_features
		self.env = env
		self.gamma = 1.#0.85
		self.alpha_r = 0.01
		self.epsilon = 0.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.99
		self.learning_rate = 0.005
		# self.tau = .125
		self.r_mean = 0.
		self.params_dict = {'delta': 0.,
							'dif': 0.,
							's2': 0.
							}

		alpha_a = 0.01 / 11.#n_features 
		alpha_c = 1.0 / 11.#n_features 
		lambda_ = 0.05
		
		self.featurizer = featurizer
		self.actor = action_model(input_shape=self.input_shape,
									alpha=alpha_a,
									gamma=self.gamma,
									lambda_=lambda_)
		
		self.critic = value_model(input_shape=self.input_shape,
									alpha=alpha_c,
									gamma=self.gamma,
									lambda_=lambda_)

	def reset(self):
		self.actor.traces = np.array([ np.zeros(self.input_shape), np.zeros(self.input_shape)])
		self.critic.traces = np.zeros(self.input_shape)

	def act(self, t_state):
		self.epsilon *= self.epsilon_decay
		self.epsilon = max(self.epsilon_min, self.epsilon)
		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()

		# already tranformed
		# t_state = self.featurizer.transform(state)
		return self.actor.predict(t_state)

	def fit(self, t_cur_state, t_new_state, reward):
		self.params_dict['delta'] = reward - self.r_mean + self.gamma * self.critic.predict(t_new_state) - self.critic.predict(t_cur_state)
		self.r_mean += self.alpha_r * self.params_dict['delta']
		self.critic.fit(t_cur_state, self.params_dict)
		self.actor.fit(t_cur_state, self.params_dict)

	def remember(self, state, action, reward, new_state, done):
		self.memory.append([state, action, reward, new_state, done])

	def save(self, name):
		aw = [self.actor.mean_layer.w,
			  self.actor.stdv_layer.w,
			  self.actor.traces
		]
		cw = [self.critic.only_layer.w,
			  self.critic.traces
		]
		f = self.featurizer
		r = self.r_mean

		os.mkdir(name)
		with open(name + "/aw.pickle", "wb") as actorfile:
			pickle.dump(aw, actorfile)
		with open(name+"/cw.pickle", "wb") as criticfile:
			pickle.dump(cw, criticfile)
		with open(name+"/feat.pickle", "wb") as featfile:
			pickle.dump(f, featfile)
		with open(name+"/r.pickle", "wb") as rfile:
			pickle.dump(r, rfile)

	def load(self, name):
		print("loading actor from {name}".format(name=name+"/aw.pickle"))
		aw = pickle.load(open(name+"/aw.pickle", "rb"))
		self.actor.mean_layer.w = aw[0]
		self.actor.stdv_layer.w = aw[1]
		self.actor.traces = aw[2]

		print("loading critic form {name}".format(name=name+"/cw.pickle"))
		cw = pickle.load(open(name+"/cw.pickle", "rb"))
		self.critic.only_layer.w = cw[0]
		self.critic.traces = cw[1]

		print("loading featurizer from {name}".format(name=name+"/feat.pickle"))
		self.featurizer = pickle.load(open(name+"/feat.pickle", "rb"))

		print("loading r_mean from {name}".format(name=name+"/r.pickle"))
		self.r_mean = pickle.load(open(name+"/r.pickle", "rb"))

		# quit()
		# n_features = pickle.load(open(name+"/n_features.pickle", "rb"))


#################################################################################################

def plot_running_avg(totalrewards):
	N = len(totalrewards)
	running_avg = np.empty(N)
	for t in range(N):
		running_avg[t] = np.array(totalrewards[max(0, t-100):(t+1)]).mean()
	plt.plot(running_avg)
	plt.title("Running Average")
	plt.show()

#################################################################################################

def plot_many_avgs(plot_rewards):
	colors = ['lightcoral',
			  'brown',
			  'red',
			  'darkorange',
			  'goldenrod',
			  'olive',
			  'yellowgreen',
			  'green',
			  'turquoise',
			  'c',
			  'deepskyblue',
			  'steelblue',
			  'navy',
			  'slateblue',
			  'blueviolet',
			  'violet',
			  'orchid',
			  'mediumvioletred']

	for i, totalrewards in plot_rewards.items():
		N = len(totalrewards)
		running_avg = np.empty(N)
		for t in range(N):
			running_avg[t] = np.array(totalrewards[max(0, t-10):(t+1)]).mean()
		plt.plot(running_avg, color=colors[int(i)])
	plt.title("Running Average")
	plt.show()





def main():
	load = False
	# load_dir = "success_-36.52746496803231"
	load_dir = "success_-1.0920613547795457"
	
	env = gym.make("MountainCarContinuous-v0")
	cur_state = env.reset().reshape(1,2)
	# print(cur_state.shape)

	# gamma   = 0.9
	# epsilon = .95

	N = 5
	plot_rewards = {}

	trials = 100
	trial_len = 5000
	shit_broke = False
	
	n_features = 1001
	# # For sklearn's Feature transformer:
	# if n_features % 4 != 0:
	# 	print("The number of features must be divisible by 4")
	# feat_components = int(n_features/4)
	feat_components = np.pi # random nonsense

	observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
	featurizer = FeatureTransformer(observation_examples, feat_components)



	for i in range(N):
		# Entire training
		cur_state = env.reset().reshape(1,2) # just for many trainings	
		env.state = np.array([-0.5,0.0])
		cur_state = env.state.reshape(1,2)
		dqn_agent = DQN(env, featurizer, n_features)

		if load:
			dqn_agent.load(load_dir)
			featurizer = dqn_agent.featurizer

		# updateTargetNetwork = 1000
		steps = []
		total_rewards = []
		max_reward = -999999.
		for trial in range(trials):
			dqn_agent.reset()
			cur_state = env.reset().reshape(1,2)
			env.state = np.array([-0.5,0.0])
			cur_state = env.state.reshape(1,2)

			t_cur_state = featurizer.transform(cur_state)
			rewards = []
			# r_mean = 0.
			for step in range(trial_len):
				# print("State:", t_cur_state)
				action = dqn_agent.act(t_cur_state)
				# print("Action:",action)
				new_state, reward, done, _ = env.step(action)
				rewards.append(reward)
				# reward = reward if not done else -20
				new_state = new_state.reshape(1,2)
				if np.isnan(new_state).any():
					shit_broke = True
					break

				t_new_state = featurizer.transform(new_state)
				dqn_agent.fit(t_cur_state, t_new_state, reward)
				dqn_agent.remember(t_cur_state, action, reward, t_new_state, done)
				# dqn_agent.replay()	   # internally iterates default (prediction) model
				# dqn_agent.target_train() # iterates target model
				t_cur_state = t_new_state
							
				if done:
					break
				
			total_reward = sum(rewards)
			total_rewards.append(total_reward)

			plot_rewards[str(i)] = total_rewards

			# if the total reward is the highest ever seen:
			if total_reward > max_reward:
				# totalrewards[max(0, t-100):(t+1)]
				max_reward = total_reward
				model_name = "./checkpoints/success_" + str(max_reward)
				dqn_agent.save(model_name)
				print("reward improved!! saving model as {name}".format(name=model_name))

			if step >= 199:
				avgrew = np.array(total_rewards[max(0, trial-100):(trial+1)]).mean()
				print("Failed to complete in trial {t} | Avg reward = {r} | Step = {step} | r_mean = {rm}".format(t=trial, r=avgrew, rm=dqn_agent.r_mean, step=step))
				if step % 10 == 0:
					print("Skipping failed model saving step")
					#dqn_agent.save_model(dqn_agent.model, "trial-{}.model".format(trial))
			else:
				if shit_broke:
					print("Broke because of Nan in trial {}".format(trial))
				else:
					print("Completed in {} trials".format(trial))
					print("Skipping successful model saving step")
				#dqn_agent.save_model(dqn_agent.model, "success.model")
				#break
		# plot_running_avg(total_rewards)
	plot_many_avgs(plot_rewards)
if __name__ == "__main__":
	# parser = argparse.ArgumentParser(description='pepar repoduction attempt')
	# parser.add_argument('direcory', type=str, help='dir for loading data')
	# parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                 const=sum, default=max,
    #                 help='sum the integers (default: find the max)')

	# args = parser.parse_args()
	main()
