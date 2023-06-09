#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import pandas as pd
import itertools
import math
import concurrent.futures
import time


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
        self.cred = {self.coinA: np.random.uniform(low=1, high=4, size=2),
                     self.coinB: np.random.uniform(low=1, high=4, size=2)}

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
    def __init__(self, w, k, epsilon=0.001, div=0.5, n_all=10):
        np.random.seed()
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
        size = round(10*self.div)
        self.subgroups = dict()
        for a in self.agents[0:size]:
            self.subgroups[a] = self.agents[0:size]
        for a in self.agents[size:10]:
            self.subgroups[a] = self.agents[size:10]
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
        n_in = {key: [] for key in self.agents}
        for a in self.agents:
            if n_in[a] == []:
                n_inA = sum([int(self.choices[b].bias==0.5) for b in self.subgroups[a]])
                n_inB = len(self.subgroups[a]) - n_inA
                for b in self.subgroups[a]:
                    n_in[b] = (n_inA, n_inB)
        for a in self.agents:
            self.choices[a] = a.choose(n_in[a][0], n_in[a][1], self.n_all)


W = [0.005, 0.0125, 0.1, 0.5, 1]
K = [0, 0.00625, 0.0125, 0.025, 0.05, 0.1]
cols = ['k', 'w', 'epsilon', 'n_all', 'div', 'alphaA', 'betaA', 'alphaB', 'betaB', 'choices']
data = pd.DataFrame(columns=cols)


def run_distrust():
    cols = ['k', 'w', 'epsilon', 'n_all', 'div', 
            'alphaA', 'betaA', 'alphaB', 'betaB', 'choices']
    df = pd.DataFrame(columns=cols)
    k = 0 # No conformity
    epsilon = 0.001 # difference between coin A and coin B biases
    n_all = 10 # agents per group
    for w in W: # levels of intergroup trust
        for div in [0.5, 0.7, 0.9]: # subgroup compositions 5-5, 7-3, 9-1
            M = Model(w=w, k=k, epsilon=epsilon, div=div, n_all=n_all)
            for t in range(3000): # 3000 time steps
                M.update()
                M.choose()
            df = df.append(pd.DataFrame([[k, 
                                          w,
                                          epsilon,
                                          n_all,
                                          div,
                                          [a.cred[M.coinA][0] for a in M.agents],
                                          [a.cred[M.coinA][1] for a in M.agents],
                                          [a.cred[M.coinB][0] for a in M.agents],
                                          [a.cred[M.coinB][1] for a in M.agents],
                                          ['A' if c.bias==0.5 else 'B' for c in list(M.choices.values())]
                                          ]], 
                                        columns=cols), 
                           ignore_index=True)
    return(df)

all_results = []
if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(run_distrust) for i in range(10000)]

        for f in concurrent.futures.as_completed(results):
            all_results.append(f.result())

data = data.append(pd.concat(all_results, ignore_index=True))
data.to_csv('wtest.csv', index=False)


def run_conformity():
    cols = ['k', 'w', 'epsilon', 'n_all', 'div', 
            'alphaA', 'betaA', 'alphaB', 'betaB', 'choices']
    df = pd.DataFrame(columns=cols)
    w = 1 # No intergroup distrust
    epsilon = 0.01 # difference between coin A and coin B biases
    n_all = 10 # agents per group
    for k in K: # levels of intergroup trust
        for div in [0.5, 0.7, 0.9, 1]: # subgroup compositions 5-5, 7-3, 9-1
            M = Model(w=w, k=k, epsilon=epsilon, div=div, n_all=n_all)
            for t in range(3000): # 3000 time steps
                M.update()
                M.choose()
            df = df.append(pd.DataFrame([[k, 
                                          w,
                                          epsilon,
                                          n_all,
                                          div,
                                          [a.cred[M.coinA][0] for a in M.agents],
                                          [a.cred[M.coinA][1] for a in M.agents],
                                          [a.cred[M.coinB][0] for a in M.agents],
                                          [a.cred[M.coinB][1] for a in M.agents],
                                          ['A' if c.bias==0.5 else 'B' for c in list(M.choices.values())]
                                          ]], 
                                        columns=cols), 
                           ignore_index=True)
    return(df)

all_results = []
if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(run_conformity) for i in range(500)]

        for f in concurrent.futures.as_completed(results):
            all_results.append(f.result())

data = data.append(pd.concat(all_results, ignore_index=True))
data.to_csv('ktest.csv', index=False)