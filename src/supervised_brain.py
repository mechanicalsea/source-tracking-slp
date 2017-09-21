"""
Supervised learner
"""


import numpy as np
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

class QLearningSupervisedNeuralNetwork(object):
	def __init__(self, actions, X, y):
		self.actions = actions  # a list
		self.Qlearner = MLPClassifier(hidden_layer_sizes=(20, 20, 10,), activation='relu', 
									  solver='adam', alpha=0.0001, batch_size='auto', 
									  learning_rate='constant', learning_rate_init=0.001, 
									  power_t=0.5, max_iter=2000, shuffle=True, random_state=None, 
									  tol=0.0001, verbose=False, warm_start=True, momentum=0.9, 
									  nesterovs_momentum=True, early_stopping=False, 
									  validation_fraction=0.1, beta_1=0.9, beta_2=0.999, 
									  epsilon=1e-08)
		X, y = shuffle(X, y)
		self.Qlearner.fit(X, y)
		print('loss: %f' % self.Qlearner.loss_)
		print(confusion_matrix(y, self.Qlearner.predict(X)))

	def choose_action(self, observation, available_actions):
		state_action = self.Qlearner.predict(observation)
		action_candidates = list(set(self.actions) & available_actions)
		action_candidates = sorted(action_candidates)
		index = np.random.permutation(action_candidates)
		state_action = state_action[index]	# some actions have same value
		action = index[state_action.round().argmax()]
		return action

	def store_transition(self, s, a, r, terminal, s_):
		pass

	def learn(self):
		pass