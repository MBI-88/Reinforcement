#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym 
import tensorflow as tf 
import numpy as np 
import random
from collections import deque,namedtuple
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(1)
np.random.seed(1)


# In[2]:


Transition=namedtuple('Transition',('state','action','reward','next_state','done'))
class DDDQN():
    def __init__(self,max_len=100,lr=0.001,gamma=0.90,epsilon_min=0.10,epsilon_deacy=0.99,epsilon_greedy=1.0,fc1=10,fc2=10,load=None):
        self.max_len=max_len
        self.lr=lr  
        self.gamma=gamma
        self.epsilon=epsilon_greedy
        self.epsilon_min=epsilon_min
        self.epsilon_deacy=epsilon_deacy
        self.input_shape=env.observation_space.shape[0]
        self.output_shape=env.action_space.n
        self.memory=deque(maxlen=self.max_len)
        self.fc1=fc1
        self.fc2=fc2
        self.counter_replay=0.
        if load==None:
            self.q_net=self.model()
            self.q_net.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss='mse')
            self.q_target=self.model()
        else:
            self.q_net=load
            self.q_target=self.model()
            self.update()
        
    
    def model(self):
        self.inputs=tf.keras.Input(shape=(self.input_shape,))
        self.h1=tf.keras.layers.Dense(units=self.fc1,activation='relu')(self.inputs)
        self.h2=tf.keras.layers.Dense(units=self.fc2,activation='relu')(self.h1)
        self.V=tf.keras.layers.Dense(units=1,activation=None)(self.h2)
        self.A=tf.keras.layers.Dense(units=self.output_shape,activation=None)(self.h2)
        self.Q=(self.V + (self.A - tf.math.reduce_mean(self.A, axis=1, keepdims=True)))
        self.modelo=tf.keras.Model(self.inputs,self.Q)
        return self.modelo
    
    def memory_saved(self,transition):
        self.memory.append(transition)
    
    def save_model(self):
        self.q_net.save('DDDQN.h5')

    def update(self):
        self.q_target.set_weights(self.q_net.get_weights())
    
    def get_qtarget(self,next_state,reward):
        action=tf.argmax(self.q_net.predict(next_state)[0]).numpy()
        q_value=self.q_target.predict(next_state)[0][action]
        q_value *= self.gamma
        q_value += reward
        return q_value
    
    def learn(self,sample):
        batch_input,batch_target=[],[]
        for trans in sample:
            state,action,reward,next_state,done=trans
            if done:
                target=reward
            else:
                target=self.get_qtarget(next_state,reward)
            target_all=self.q_net.predict(state)[0]
            target_all[action]=target
            batch_target.append(target_all)
            batch_input.append(state.flatten())
            self.ajust_epsilon()
        return self.q_net.train_on_batch(x=np.array(batch_input),y=np.array(batch_target))

    def ajust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_deacy

    def replay(self,sample_memory):
        sample=random.sample(self.memory,sample_memory)
        hist=self.learn(sample)
        if self.counter_replay % 10 ==0:
            self.update()
        self.counter_replay += 1
    
    def choose_action(self,state):
        if np.random.rand() <= self.epsilon: # Exploracion
            return random.randrange(0,self.output_shape)
        Q_value=self.q_net.predict(state)[0] # Explotacion
        return tf.argmax(Q_value).numpy()


# In[3]:


episodes=10000
max_len=10000
epsilon_decay=0.99
epsilo_min=0.10
gamma=0.999
fc_1=100
fc_2=100
learning_rate=0.001
batch_size=32


try:
    load=tf.keras.models.load_model('DDDQN.h5')
except:
    load=None
env=gym.make('MountainCar-v0')
agente=DDDQN(max_len=max_len,epsilon_deacy=epsilon_decay,epsilon_min=epsilo_min,fc1=fc_1,fc2=fc_2,lr=learning_rate,gamma=gamma,load=load)

state=env.reset()
state=np.reshape(state,[1,state.shape[0]])

for i in range(1000):
    action=agente.choose_action(state)
    next_state,reward,done,_=env.step(action)
    next_state=np.reshape(next_state,[1,next_state.shape[0]])
    agente.memory_saved(Transition(state,action,reward,next_state,done))
    if done:
        state=env.reset()
        state=np.reshape(state,[1,state.shape[0]])
    else:
        state=next_state

print('Memory full')
premios=0.
for o in range(1,episodes):
    state=env.reset()
    state=np.reshape(state,[1,state.shape[0]])
    while True:
        action=agente.choose_action(state)
        next_state,reward,done,_=env.step(action)
        next_state=np.reshape(next_state,[1,next_state.shape[0]])
        agente.memory_saved(Transition(state,action,reward,next_state,done))
        premios += reward
        env.render()
        if done:
            print('Total Rewards: {} Episodes: {}/{} '.format(premios,o,episodes))
            agente.save_model()
            break
        state=next_state
        agente.replay(batch_size)
env.close()

