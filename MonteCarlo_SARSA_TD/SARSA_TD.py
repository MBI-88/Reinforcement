#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gym 
import numpy as np 
from collections import defaultdict
import time
from IPython.display import clear_output
np.random.seed(10)


# In[ ]:


env=gym.make('Taxi-v3')


# ***Metodo SARSA***

# In[ ]:


Q_sarsa=defaultdict(lambda:np.zeros(env.action_space.n))
alpha = 0.85
gamma = 0.9
def epsilon_greddy(state):
     if np.random.uniform() < 0.10:
         return env.action_space.sample()
     else:
         q_value=Q_sarsa[state]
         permutation=np.random.permutation(env.action_space.n)
         q_value=[q_value[a] for a in permutation]
         permutation_max=np.argmax(q_value)
         action=permutation[permutation_max]
         return action

def update_Q(state,action,nextaction,reward,nextstate):
    Q_sarsa[state][action] += alpha *(reward + gamma*Q_sarsa[nextstate][nextaction] - Q_sarsa[state][action])

def train(politica,env):
    state=env.reset()
    action=politica(state)
    r=0.0
    while True:
        nextstate,reward,done,_=env.step(action)
        nextaction=politica(nextstate)
        update_Q(state,action,nextaction,reward,nextstate)
        env.render()
        time.sleep(1.1)
        clear_output()
        action=nextaction
        state=nextstate
        r += reward
        if done:
            print('Premios obtenidos: ',r)
            time.sleep(1)
            break
        


# In[ ]:


for i in range(100):
     train(epsilon_greddy,env)
env.close()


# ***Metodo de TD***

# In[ ]:


Q_td=defaultdict(lambda:np.zeros(env.action_space.n))
alpha = 0.85
gamma = 0.9

def epsilon_greddy(state):
     if np.random.uniform() < 0.10:
         return env.action_space.sample()
     else:
         q_value=Q_td[state]
         permutation=np.random.permutation(env.action_space.n)
         q_value=[q_value[a] for a in permutation]
         permutation_max=np.argmax(q_value)
         action=permutation[permutation_max]
         return action

def update_Q(state,action,reward,nextstate):
    Q_td[state][action] += alpha *(reward + gamma*np.amax(Q_td[nextstate]) - Q_td[state][action])

def train(politica,env):
    state=env.reset()
    r=0.0
    while True:
        action=politica(state)
        nextstate,reward,done,_=env.step(action)
        update_Q(state,action,reward,nextstate)
        env.render()
        time.sleep(1.1)
        clear_output()
        state=nextstate
        r += reward
        if done:
            print('Premios obtenidos: ',r)
            time.sleep(1)
            break


for i in range(100):
     train(epsilon_greddy,env)
env.close()

