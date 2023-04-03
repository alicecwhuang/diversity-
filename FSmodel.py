#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import pandas as pd
import itertools
import math


# In[10]:


class Coin:
    def __init__(self, bias=0.5):
        self.bias = bias
    def toss(self, n=1000):
        return np.random.binomial(n, self.bias, size=None)


# In[11]:


class Agent:
    def __init__(self, no, coinA, coinB, w=0.5, k=0.5):
        self.no = no
        self.w = w # intergroup distrust
        self.k = k # conformity
        self.coinA = coinA
        self.coinB = coinB
        """Credence is modelled by beta distribution. Record parameters alpha and beta for each coin."""
        self.cred = {self.coinA: np.array(random.choices(range(1, 5), k=2)), 
                     self.coinB: np.array(random.choices(range(1, 5), k=2))}
    def update(self, coin, data):
        """data in the form [head, tail]"""
        self.cred[coin] = np.sum([self.cred[coin], data], axis=0)
    def exputil(self, alpha, beta, n_in, n_all):
        """n_in: number of in-group members choosing that coin"""
        """n_all: total number of members in network"""
        p = alpha/(alpha + beta)
        return ((1-self.k)*p + self.k*n_in/n_all)
    def choose(self, n_inA, n_inB, n_all):
        """choose which coin to flip based on credence and peer influence"""
        euA = self.exputil(self.cred[self.coinA][0], self.cred[self.coinA][1], n_inA, n_all)
        euB = self.exputil(self.cred[self.coinB][0], self.cred[self.coinB][1], n_inB, n_all)
        if euA > euB:
            return self.coinA
        else:
            return self.coinB


# In[12]:


class Model:
    def __init__(self, w=0.5, k=0.5, epsilon=0.01, div=0.5, n_all=8):
        self.w = w # intergroup distrust
        self.k = k # conformity
        self.epsilon = epsilon # difference between coins (difficulty)
        self.div = div # subgroup size
        self.n_all = n_all # total number of agents
        self.subgroups = dict()
        """create coins"""
        self.coinA = Coin()
        self.coinB = Coin(0.5+epsilon)
        """create agents"""
        self.agents = []
        for i in range(n_all):
            self.agents.append(Agent(i, self.coinA, self.coinB, self.w, self.k))
        """divide agents into subgroups"""
        """parameter div specifies largest subgroup size"""
        """e.g. div=0.7 means 7-3 (2 subgroups) and div=0.4 means 4-4-2 (2 subgroups)"""
        for a in self.agents:
            self.subgroups[a] = [a]
        if self.div > 0:
        """if div=0 no other ingroup member other than self"""
            M = self.agents
            remaining = n_all
            size = math.ceil(n_all*self.div)
            while remaining > size:
                subgroup = random.choices(M, k=size)
                for m in subgroup:
                    self.subgroups[m] = subgroup
                remaining -= size
                M = [x for x in M if x not in subgroup]
            for m in M:
                self.subgroups[m] = M
        """record choices of which coin to flip"""
        self.choices= dict()
        for a in self.agents:
            self.choices[a] = a.choose(0, 0, self.n_all) # no peer influence at the start
        
    def update(self, n_flips=1000):
        """agents flip coin one by one"""
        for a in self.agents:
            coin = self.choices[a]
            result = coin.toss(n=n_flips) # 1000 flips per time step
            data = np.array([result, n_flips-result])
            """each time an agent flips, they share data with everyone (complete network)"""
            for b in self.agents:
                if b in self.subgroups[a]:
                    b.update(coin, data) # update on ingroup data (including own data)
                else:
                    b.update(coin, data*self.w) # update on outgroup data (reduced trust)
                    
    def choose(self):
        """all agents choose which coin to flip in the next round"""
        for a in self.agents:
            n_inA = 0
            n_inB = 0
            for b in self.subgroups[a]:
                if self.choices[b].bias == 0.5:
                    n_inA += 1
                else:
                    n_inB += 1
            self.choices[a] = a.choose(n_inA, n_inB, self.n_all)


# In[13]:


cols = ['k', 'w', 'epsilon', 'n_all', 'div', 
        'alphaA', 'betaA', 'alphaB', 'betaB', 'choices']
df = pd.DataFrame(columns=cols)


# In[35]:


runs_per_config = 400

"""Distrust"""
for run in range(runs_per_config):
    for k in [0]: # set k=0. No conformity.
        for w in [0, 0.005, 0.0125, 0.1, 0.5, 1]: # levels of intergroup trust
            for epsilon in [0.001]: # difference between coin A and coin B biases
                for n_all in [10]: # agents per group
                    for div in [0.5, 0.7, 0.9]: # subgroup compositions 5-5, 7-3, 9-1
                        M = Model(w=w, k=k, epsilon=epsilon, div=div, n_all=n_all)
                        for t in range(3000): # 3000 time steps
                            M.update()
                            M.choose()
                            alphaA = [a.cred[M.coinA][0] for a in M.agents]
                            betaA = [a.cred[M.coinA][1] for a in M.agents]
                            alphaB = [a.cred[M.coinB][0] for a in M.agents]
                            betaB = [a.cred[M.coinB][1] for a in M.agents]
                            choices = ['A' if c.bias==0.5 else 'B' for c in list(M.choices.values())]
                        df = df.append(pd.DataFrame([[k, 
                                                      w,
                                                      epsilon,
                                                      n_all,
                                                      div,
                                                      alphaA,
                                                      betaA,
                                                      alphaB,
                                                      betaB,
                                                      choices]], 
                                                    columns=cols), 
                                       ignore_index=True)
                        df.to_csv('test.csv', index=False)


"""Conformity""" 
for run in range(runs_per_config):
    for k in [0.01, 0.02, 0.05, 0.1]: # levels of conformity 
        for w in [1]: # set w=1. No intergroup distrust
            for epsilon in [0.001]:
                for n_all in [10]:
                    for div in [0.5, 1]: # subgroup compositions 5-5 and 10-0
                        M = Model(w=w, k=k, epsilon=epsilon, div=div, n_all=n_all)
                        for t in range(3000):
                            M.update()
                            M.choose()
                            alphaA = [a.cred[M.coinA][0] for a in M.agents]
                            betaA = [a.cred[M.coinA][1] for a in M.agents]
                            alphaB = [a.cred[M.coinB][0] for a in M.agents]
                            betaB = [a.cred[M.coinB][1] for a in M.agents]
                            choices = ['A' if c.bias==0.5 else 'B' for c in list(M.choices.values())]
                        df = df.append(pd.DataFrame([[k, 
                                                      w,
                                                      epsilon,
                                                      n_all,
                                                      div,
                                                      alphaA,
                                                      betaA,
                                                      alphaB,
                                                      betaB,
                                                      choices]], 
                                                    columns=cols), 
                                       ignore_index=True)
                        df.to_csv('test.csv', index=False)




