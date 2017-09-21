"""
observation: {0:o, 1:o, 2:o, 3:o} # up, down, left, right
Position: [int, int]
"""

import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.cluster import MiniBatchKMeans

np.random.seed(42)

UNIT = 40   # pixels
FAKE_PATH = 'dataset/fake_observation.csv' 
NORMAL_PAYH = 'dataset/normal_observation.csv'
FEATURES = [
    'component_0',
    'component_1'
]


class System(object):
    """
    CPS Network:
    _build_maze(self): construct a network as well as backbones and fake nodes
    reset(self): reset attacker location (initial location) and get observation
    setp(self, action): moving, observe, immediate reward
    _move(self, base_action): implement movement
    get_availabel_director(self): get availabel director
    _get_neighboring_observation(self): get observation [n_action, n_features]
    """
    def __init__(self, system_h=10, system_w=10, system_fake=0.1):
        self.system_h = system_h       # grid height
        self.system_w = system_w       # grid width
        self.system_fake = system_fake # fake nodes: percent of total nodes 

        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)

        self.title = 'system'
        self._build_observation()
        self._build_maze()

    def _build_observation(self):
        # load fake and normal observation
        self.fake_observations = pd.read_csv(FAKE_PATH)[FEATURES].values
        self.normal_observations = pd.read_csv(NORMAL_PAYH)[FEATURES].values
        self.n_features = self.fake_observations.shape[1] * self.n_actions

    def _get_single_observation(self, i, normal):
        if normal:
            x = np.random.choice(self.normal_index_labels[i])
            return self.normal_observations[x]
        else:
            x = np.random.choice(self.fake_index_labels[i])
            return self.fake_observations[x]    

    def _get_position(self, vi):
        if vi in self.backbones:
            return self.normal_observations[self.backbones2observations[vi]]# + np.random.normal(0, 0.01, 2)
        else:
            return self.fake_observations[self.fakes2observations[vi]]# + np.random.normal(0, 0.01, 2)
        # w = vi % self.system_w
        # h = int(vi / self.system_w)
        # return np.array([w-self.system_w+1, h-self.system_h+1])*1.0/self.system_w + np.random.normal(0, 0.1, 2)

    def _build_maze(self):
        # create grids nodes: x, y, s
        self.nodes = []
        for h in range(self.system_h):
            for w in range(self.system_w):
                x, y, s = h * UNIT, w * UNIT, h * self.system_w + w
                self.nodes.append([x, y, s])
        # create oval
        self.oval = np.array([UNIT*(self.system_h-1), UNIT*(self.system_w-1)])  # s = self.system_h * self.system_w - 1
        # create red rect
        self.rect = np.array([0, 0])                                  # s = 0
        # create random path from initial rect -> oval: down and right
        self.nodes_adj = {}
        self.backbones = set()
        u = 0
        while True:
            self.backbones.add(u)
            if u == self.system_h * self.system_w - 1:
                break
            director = np.round(np.random.rand()).astype(int)
            if director == 0:   # right
                if u % self.system_w == self.system_w - 1:
                    v = u + self.system_w
                else:
                    v = u + 1
            elif director == 1: # down
                if u >= self.system_w * (self.system_h - 1):
                    v = u + 1
                else:
                    v = u + self.system_w
            # u and v add neighbors
            if u == 0:
                self.nodes_adj[u] = set([v])
            else:
                self.nodes_adj[u].add(v)
            self.nodes_adj[v] = set([u])
            # move u to v
            u = v
        # create fake nodes and it add neighbors
        fakes = list(set(range(self.system_h*self.system_w)) - self.backbones)
        np.random.shuffle(fakes)
        self.fake_nodes = set(fakes[:int(self.system_h * self.system_w * self.system_fake)])
        for fn in self.fake_nodes:
            if fn == 0:
                v = [fn+1, fn+self.system_w]
            elif fn == self.system_w - 1:
                v = [fn-1, fn+self.system_w]
            elif fn == self.system_w * (self.system_h-1):
                v = [fn-self.system_w, fn+1]
            elif fn == self.system_w * self.system_h - 1:
                v = [fn-self.system_w, fn-1]
            elif fn < self.system_w:
                v = [fn-1, fn+1, fn+self.system_w]
            elif fn >= self.system_w * (self.system_h-1):
                v = [fn-1, fn+1, fn-self.system_w]
            elif fn % self.system_w == 0:
                v = [fn+1, fn-self.system_w, fn+self.system_w]
            elif fn % self.system_w == self.system_w - 1:
                v = [fn-1, fn-self.system_w, fn+self.system_w]
            else:
                v = [fn-1, fn+1, fn-self.system_w, fn+self.system_w]
            self.nodes_adj[fn] = set(v)
            for vi in v:
                if self.nodes_adj.get(vi):
                    self.nodes_adj[vi].add(fn)
                else:
                    self.nodes_adj[vi] = set([fn])
        
        self.backbones = sorted(list(self.backbones))
        self.fake_nodes = sorted(list(self.fake_nodes))

        self.backbones2observations = {}
        self.fakes2observations = {}
        bo_seq = np.random.permutation(len(self.backbones))
        fo_seq = np.random.permutation(len(self.fake_nodes))
        for i in range(len(self.backbones)):
            self.backbones2observations[self.backbones[i]] = bo_seq[i]
        for i in range(len(self.fake_nodes)):
            self.fakes2observations[self.fake_nodes[i]] = fo_seq[i]

        print('backbone:', self.backbones)
        print('fake node:', self.fake_nodes)

    def reset(self):
        self.rect = np.array([0, 0])
        observation = self._get_neighboring_observation()
        # return observation
        return observation
    
    def step(self, action):
        s = self.rect
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > 0:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (self.system_h - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # left
            if s[0] > 0:
                base_action[0] -= UNIT
        elif action == 3:   # right
            if s[0] < (self.system_w - 1) * UNIT:
                base_action[0] += UNIT
        
        self.rect = self._move(base_action)  # move agent

        s_ = self._get_neighboring_observation()  # next state/observation

        # reward function
        if self.rect[0] == self.oval[0] and self.rect[1] == self.oval[1]:
            reward = 0.0
            done = True
        else:
            reward = -1.0
            done = False

        return s_, reward, done
    
    def _move(self, base_action):
        u = (self.rect/UNIT).astype(int)
        pos_u = u[0] + u[1] * self.system_w
        v = (self.rect/UNIT).astype(int) + (base_action/UNIT).astype(int)
        pos_v = v[0] + v[1] * self.system_w
        # return next state
        if pos_v in self.nodes_adj.get(pos_u) \
            and (pos_v in self.fake_nodes or pos_v in self.backbones):
            return v * UNIT
        else:
            return u * UNIT
    
    def get_availabel_director(self):
        """disregarded"""
        u = (self.rect/UNIT).astype(int)
        pos_u = u[0] + u[1] * self.system_w
        directors = []
        for i, vi in enumerate([pos_u-self.system_w, pos_u+self.system_w, pos_u-1, pos_u+1]): # up, down, left, right
            if vi in self.nodes_adj[pos_u] and (vi in self.backbones or vi in self.fake_nodes):
                directors.append(i)
        return directors
    
    def _get_neighboring_observation(self):
        u = (self.rect/UNIT).astype(int)
        pos_u = u[0] + u[1] * self.system_w
        # return self._get_position(pos_u)
        observation = []      # [n_actions, n_features,] => up, down, left, right
        for i, vi in enumerate([pos_u-self.system_w, pos_u+self.system_w, pos_u-1, pos_u+1]): # up, down, left, right
            if vi in self.nodes_adj[pos_u]:
                if vi in self.fake_nodes:
                    # observation.append(self._get_single_observation(vi, False))
                    observation.append(self._get_position(vi))
                elif vi in self.backbones:
                    # observation.append(self._get_single_observation(vi, True))
                    observation.append(self._get_position(vi))
                else:
                    zeros = np.zeros(int(self.n_features/self.n_actions))
                    observation.append(zeros)
            else:
                zeros = np.zeros(int(self.n_features/self.n_actions))
                observation.append(zeros)
        # return neighboring observation or available actions
        return np.hstack(observation)

    def render(self):
        pass
    
    def destroy(self):
        """set up a new environment"""
        self.backbones2observations = {}
        self.fakes2observations = {}
        bo_seq = np.random.permutation(len(self.backbones))
        fo_seq = np.random.permutation(len(self.fake_nodes))
        for i in range(len(self.backbones)):
            self.backbones2observations[self.backbones[i]] = bo_seq[i]
        for i in range(len(self.fake_nodes)):
            self.fakes2observations[self.fake_nodes[i]] = fo_seq[i]

    