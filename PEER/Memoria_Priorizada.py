#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
from collections import deque,namedtuple
import tensorflow as tf 
import random,os
import gym
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(1)
np.random.seed(1)


# In[2]:


class PriorizedMemory():
    def __init__(self,maxlen=100):
        self.memory=deque(maxlen=maxlen)
        self.priorities=deque(maxlen=maxlen)

    def add(self,experience):
        self.memory.append(experience)
        self.priorities.append(max(self.priorities,default=1))
    
    def get_probabilities(self,priorities_scale):
        scaled_priorities=np.array(self.priorities)**priorities_scale
        sample_probabilities=scaled_priorities/sum(scaled_priorities)
        return sample_probabilities
    
    def get_importances(self,probabilities):
        importance=1/len(self.memory)+1/probabilities
        importance_nor=importance/max(importance)
        return importance_nor
    
    def sample(self,batch_size,priority_scale=1.0):
        sample_size=min(len(self.memory),batch_size)
        sample_prob=self.get_probabilities(priority_scale)
        sample_indx=random.choices(range(len(self.memory)),k=sample_size,weights=sample_prob)
        samples=np.array(self.memory)[sample_indx]
        importance=self.get_importances(sample_prob[sample_indx])
        return samples,sample_indx,importance

    def set_priorities(self,indices,errors,offset=0.1):
        for i,e in zip(indices,errors):
            self.priorities[i]= e + offset


# In[3]:


class DDQN():
    def __init__(self,maxlen=100,gamma=0.95,epsilon_min=0.10,epsilon_greddy=1.,epsilon_decay=0.99,load=False,lr=0.001):
        self.buffer=PriorizedMemory(maxlen=maxlen)
        self.gamma=gamma
        self.counter=0.0
        self.epsilon_decay=epsilon_decay
        self.epsilon_min=epsilon_min
        self.epsilon=epsilon_greddy
        self.lr=lr
        self.input_shape=env.observation_space.shape[0]
        self.output_shape=env.action_space.n
        self.load=load
        self.importance=0
       
        
        
        if not load:
            self.q_net=self.make_model()
            self.q_target=self.make_model()
        else:
            self.q_net=self.make_model()
            self.q_target=self.make_model()
            self.q_net.load_weights('DDQN_PRE.h5')
            print('Modelo cargado...')
        
        self.update_weight()
    

    def new_losses(self,importance):
        def losses(y_true,y_pred):
            return tf.reduce_mean(tf.multiply(tf.square(y_true - y_pred),self.importance))
        return losses

    def make_model(self):
        self.model=tf.keras.Sequential([
            tf.keras.layers.Dense(units=100,activation='relu',input_shape=(self.input_shape,)),
            tf.keras.layers.Dense(units=100,activation='relu'),
            tf.keras.layers.Dense(units=50,activation='relu'),
            tf.keras.layers.Dense(units=self.output_shape)
        ])
        losses=self.new_losses(self.importance)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),loss='mse')
        return self.model
        
    def save_model(self):
        self.q_net.save_weights('DDQN_PRE.h5')     

    def update_weight(self):
        self.q_target.set_weights(self.q_net.get_weights())

    def get_Qtarget(self,next_state,reward):
        next_state=tf.expand_dims(tf.convert_to_tensor(next_state),0)
        action=np.argmax(self.q_net.predict(next_state)[0])
        q_value=self.q_target.predict(next_state)[0][action]
        q_value *= self.gamma
        q_value += reward
        return q_value

    
    def _learn(self,batch_sample,priority=1.0):
        muestra=self.buffer.sample(batch_sample,priority)
        batch_state,batch_target,target_old=[],[],[],
        for tran in muestra[0]:
            s,a,r,next_s,done=tran
            if done:
                target=r
            else:
                target=self.get_Qtarget(next_s,r)

            target_all=self.q_net.predict(tf.expand_dims(tf.convert_to_tensor(s),0))[0]
            target_old.append(target_all.copy())
            target_all[a]=target
            batch_state.append(s)
            batch_target.append(target_all) 
        self.update_epsilon()
        self.counter += 1 
        if self.counter % 10 == 0:
            self.update_weight()

        state_batch=tf.convert_to_tensor(batch_state)
        target_batch=tf.convert_to_tensor(batch_target)
        self.importance=tf.cast(tf.convert_to_tensor(muestra[2]),dtype=tf.float32)
        self.importance=tf.expand_dims(self.importance,axis=1)

        y_pred=np.array(target_old)
        y_true=np.array(batch_target)
        index=np.arange(min(len(self.buffer.memory),batch_sample),dtype=np.int32)
        error= np.abs(y_true[index,np.array(action)] - y_pred[index,np.array(action)])
        self.buffer.set_priorities(muestra[1],error)
        return self.q_net.train_on_batch(x=state_batch,y=target_batch)
        

    def update_epsilon(self):
        if self.load:
            self.epsilon = self.epsilon_min
        else:
            if self.epsilon >= self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def choose_action(self,state):
        if np.random.rand() <= self.epsilon: # Exploracion
            return random.randrange(0,self.output_shape)
        state=tf.expand_dims(tf.convert_to_tensor(state),0)
        Q_value=tf.argmax(self.q_net.predict(state)[0]).numpy() # Explotacion
        return Q_value


# In[4]:


episode=10000
gamma=0.999
epsilon_min=0.10
max_len=10000
lr=0.001
batch_size=32
priority=1.0
Transition=namedtuple('Transition',('state','action','reward','next_state','done'))

if os.path.exists('DDQN_PRE.h5'):
    load=True
else: load=False

env=gym.make('CartPole-v1')
env.metadata['video.frames_per_second']=60
agent=DDQN(maxlen=max_len,lr=lr,epsilon_min=epsilon_min,load=load)
state=env.reset()
for i in range(episode):
    action=agent.choose_action(state)
    next_state,reward,done,_=env.step(action)
    agent.buffer.add(Transition(state,action,reward,next_state,done))
    if done:
        state=env.reset()
    else:
        state=next_state
print('Memory Full')
total_r=0.0
history_loss=[]
for i in range(1,episode):
    state=env.reset()
    while True:
        action=agent.choose_action(state)
        next_state,reward,done,_=env.step(action)
        agent.buffer.add(Transition(state,action,reward,next_state,done))
        total_r += reward
        #env.render()
        if done:
            print('Total Rewards: {} Episodes: {}/{} Epsilon: {}'.format(total_r,i,episode,agent.epsilon))
            if i % 20 == 0:
                agent.save_model()
            break
        loss=agent._learn(batch_size,priority)
        history_loss.append(loss)
        state=next_state
env.close()

