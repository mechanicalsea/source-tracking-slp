"""
Random Walk
"""


import numpy as np

class RandomWalk(object):
	def __init__(self, actions):
		self.actions = actions  # a list

	def choose_action(self, observation, available_actions):
		action = np.random.choice(list(set(self.actions) & available_actions))
		return action

	def store_transition(self, s, a, r, terminal, s_):
		pass

	def learn(self):
		pass