#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import gym
from  collections import defaultdict
np.random.seed(42)


# In[2]:


env= gym.make('Blackjack-v0')


# In[3]:


env.action_space,env.observation_space,env.reward_range,env.player,env.natural,env.np_random,env.spec


# In[4]:


lista_victorias=[]
lista_derrotas=[]
lista_empates=[]
q_table=defaultdict(lambda : np.zeros(env.action_space.n))

def epsilon_greddy_policy(observation):
    if np.random.uniform() < 0.10:
        return env.action_space.sample()
    else:
        q_value=q_table[observation]
        permutation=np.random.permutation(env.action_space.n)
        q_value=[q_value[a] for a in permutation]
        permutation_max=np.argmax(q_value)
        action=permutation[permutation_max]
        return action

def sample_policy(observation):
    score,dealer,ace=observation
    return 0 if  score >= 20 else  1

def train(policy,env,victoria,derrota,empates):
    states,actions,rewards = [],[],[]
    observation = env.reset()
    while True:
        states.append(observation)
        action = policy(observation)
        actions.append(action)
        observation,reward,done,info = env.step(action)
        rewards.append(reward)
        if reward == 1:
            lista_victorias.append(+1)
        elif reward==0:
            lista_empates.append(+1)
        else:
            lista_derrotas.append(+1)

        if done:
            break
    return states,actions,rewards

def first_visit_mc_prediction(policy,env,n_episodes):
    value_table=defaultdict(float)
    N=defaultdict(int)
    for _ in range(n_episodes):
        states,actions,rewards=train(epsilon_greddy_policy,env,lista_victorias,lista_derrotas,lista_empates)
        returns=0
        for t in range(len(states)-1,-1,-1):
            R=rewards[t]
            S=states[t]
            A=actions[t]
            returns += R
            if S not in states[:t]:
                N[S] += 1
                value_table[S] += (returns - value_table[S])/N[S] # Funcion de actualizacion de la politica
                q_table[S][A] = value_table[S]
    print('Victorias {} Derrotas {} Empates {}'.format(len(lista_victorias),len(lista_derrotas),len(lista_empates)))


# In[5]:


first_visit_mc_prediction(epsilon_greddy_policy,env,10000)


# In[ ]:




