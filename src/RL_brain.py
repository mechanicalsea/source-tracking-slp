"""
class:
	DeepQNetwork
	QLearningTable
"""

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
	def __init__(
			self,
			n_actions,
			n_features,
			learning_rate=0.01,
			reward_decay=0.9,
			epsilon=0.9,
			replace_target_iter=300,
			memory_size=500,
			batch_size=32,
			e_greedy_increment=None,
			output_graph=False
	):
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon_max = epsilon
		self.replace_target_iter = replace_target_iter
		self.memory_size = memory_size
		self.batch_size = batch_size
		self.epsilon_increment = e_greedy_increment
		self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

		# total learning step
		self.learn_step_counter = 0

		# initialize zero memory [s, a, r, s_]
		self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
		self.memory_counter = 0

		# consist of [target_net, evaluate_net]
		self._build_net()
		t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
		e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
		self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

		self.sess = tf.Session()

		if output_graph:
			# $ tensorboard --logdir=logs
			# tf.train.SummaryWriter soon be deprecated, use following
			tf.summary.FileWriter("logs/", self.sess.graph)

		self.sess.run(tf.global_variables_initializer())
		self.cost_his = []

	def _build_net(self):
		# ------------------ all inputs ------------------------
		self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
		self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
		self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
		self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
		# self.terminals = tf.placeholder(tf.float32, [None, ], name='terminals') # terminals: 0/1 - terminal/not terminal

		w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
		# ------------------ build evaluate_net ------------------
		with tf.variable_scope('eval_net'):
			e1 = tf.layers.dense(self.s, 512, tf.nn.relu, kernel_initializer=w_initializer,
								 bias_initializer=b_initializer)
			# e2 = tf.layers.dense(e1, 50, tf.nn.relu, kernel_initializer=w_initializer,
			# 					 bias_initializer=b_initializer)
			# e3 = tf.layers.dense(e2, 25, tf.nn.relu, kernel_initializer=w_initializer,
			# 					 bias_initializer=b_initializer)
			self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
										  bias_initializer=b_initializer)

		# ------------------ build target_net ------------------
		with tf.variable_scope('target_net'):
			t1 = tf.layers.dense(self.s_, 512, tf.nn.relu, kernel_initializer=w_initializer,
								 bias_initializer=b_initializer)
			# t2 = tf.layers.dense(t1, 50, tf.nn.relu, kernel_initializer=w_initializer,
			# 					 bias_initializer=b_initializer)
			# t3 = tf.layers.dense(t2, 25, tf.nn.relu, kernel_initializer=w_initializer,
			# 					 bias_initializer=b_initializer)
			self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
										  bias_initializer=b_initializer)

		with tf.variable_scope('q_target'):
			q_target = self.r + \
					   self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')# * self.terminals
			self.q_target = tf.stop_gradient(q_target)
		with tf.variable_scope('q_eval'):
			a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
			self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)	# shape=(None, )
		with tf.variable_scope('loss'):
			self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
		with tf.variable_scope('train'):
			self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

	def reset(self):
		self.sess.run(tf.global_variables_initializer())
		self.epsilon = 0 if self.epsilon_increment is not None else self.epsilon_max
		self.learn_step_counter = 0
		self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))
		self.memory_counter = 0
		self.cost_his = []

	def store_transition(self, s, a, r, s_):
		transition = np.hstack((s, [a, r], s_))
		# replace the old memory with new memory
		index = self.memory_counter % self.memory_size
		self.memory[index, :] = transition
		self.memory_counter += 1

	def choose_action(self, observation):
		# to have batch dimension when feed into tf placeholder
		observation = observation[np.newaxis, :]

		if np.random.uniform() < self.epsilon:
			# forward feed the observation and get q value for every actions
			actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
			action = np.argmax(actions_value)
		else:
			action = np.random.randint(0, self.n_actions)
		return action

	# def choose_action(self, observation, available_director, director_proba=None):
	# 	if np.random.uniform() > self.epsilon:
	# 		return np.random.choice(available_director)
	# 	# to have batch dimension when feed into tf placeholder
	# 	observation = observation[np.newaxis, :]
	# 	actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})[0]
	# 	available_director = np.random.permutation(available_director)
	# 	# forward feed the observation and get q value for every actions
	# 	if director_proba is not None: # predicted transition
	# 		director_proba = director_proba[available_director]
	# 		director_proba /= director_proba.sum()
	# 		actions_value = actions_value[available_director]
	# 		actions_value *= director_proba
	# 		return available_director[np.argmax(actions_value)]
	# 	else: # no predicted transition
	# 		return available_director[np.argmax(actions_value[available_director])]

	def learn(self):
		# check to replace target parameters, update q_next network
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.sess.run(self.target_replace_op)

		# sample batch memory from all memory
		if self.memory_counter > self.memory_size:
			sample_index = np.random.choice(self.memory_size, size=self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
		batch_memory = self.memory[sample_index, :]

		_, self.cost = self.sess.run(
			[self._train_op, self.loss],
			feed_dict={
				self.s: batch_memory[:, :self.n_features],  # newest params
				self.a: batch_memory[:, self.n_features],
				self.r: batch_memory[:, self.n_features+1],
				self.s_: batch_memory[:, -self.n_features:],  # fixed params
			})

		self.cost_his.append(self.cost)

		# increasing epsilon
		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
		self.learn_step_counter += 1

class QLearningTable:
	def __init__(self, actions, learning_rate=0.1, reward_decay=0.9, epsilon=0.1):
		self.actions = actions  # a list
		self.n_actions = len(actions)
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = epsilon
		self.q_table = {}

	def store_transition(self, s, a, r, done, s_):
		pass

	def choose_action(self, observation, available_director, director_proba=None):
		observation = self._ndarray2str(observation)
		self.check_state_exist(observation)
		# action selection
		if type(director_proba) == np.ndarray: # predicted transition
			if np.random.uniform() < self.epsilon: # explore
				return np.random.choice(available_director)
			director_proba = np.array(director_proba)
			# choose best action
			state_action = self.q_table[observation]
			# some actions have same value
			available_director = np.random.permutation(available_director)
			state_action = state_action[available_director]
			state_action[state_action == 0.0] = np.finfo(float).eps  # 处理 value 为 0 的值为最小正数

			director_proba = director_proba[available_director]
			director_proba /= director_proba.sum()

			state_action *= director_proba
			# best action
			action = available_director[state_action.argmax()]
			return action
		else: # no predicted transition
			director_proba = np.zeros(self.n_actions)
			director_proba[available_director] += (self.epsilon / len(available_director))
			# choose best action
			state_action = self.q_table[observation]
			# some actions have same value
			available_director = np.random.permutation(available_director)
			state_action = state_action[available_director]
			# best action
			action = available_director[state_action.argmax()]
			# give probability 1.0-epsilon to best action
			director_proba[action] += (1.0 - self.epsilon)
			return np.random.choice(self.actions, p=director_proba)
		
	def learn(self, s, a, r, done, s_):
		s, s_ = self._ndarray2str(s), self._ndarray2str(s_)
		self.check_state_exist(s_)
		q_predict = self.q_table[s][a]
		if not done:
			q_target = r + self.gamma * self.q_table[s_].max()  # next state is not terminal
		else:
			q_target = r  # next state is terminal
		self.q_table[s][a] += self.lr * (q_target - q_predict)  # update

	def _ndarray2str(self, s):
		return ','.join(['%.6f' % x if x != 0. else str(0) for x in s])

	def check_state_exist(self, state):
		if state not in self.q_table.keys():
			# append new state to q table
			self.q_table[state] = np.zeros(self.n_actions)

	def reset(self):
		self.q_table = {}