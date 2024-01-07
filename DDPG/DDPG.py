#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf  
import gym,time,random
from collections import deque,namedtuple 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(1)


# In[2]:


Tupla=namedtuple('Tupla',('state','action','reward','next_state','done'))
class DDPG():
    def __init__(self,tau=1.0,gamma=0.97,max_len=1000,load=False):
        self.memory=deque(maxlen=max_len)
        self.gamma=gamma
        self.load=load
        self.action_dim=env.action_space.shape[0]
        self.state_dim=env.observation_space.shape[0]
        self.action_low=env.action_space.low[0]
        self.action_high=env.action_space.high[0]
        self.tau=tau
        self.actor_optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer=tf.keras.optimizers.Adam(learning_rate=0.002)
        if self.load:
            self.target_actor=self.actor()
            self.target_critic=self.critico()
            self.network_actor=tf.keras.models.load_model('Actor_ddpg.h5')
            time.sleep(2)
            self.network_critic=tf.keras.models.load_model('Critic_ddpg.h5')
        else:
            self.network_actor=self.actor()
            self.network_critic=self.critico()
            self.target_actor=self.actor()
            self.target_critic=self.critico()
        self.update_networks()
        
    
    def remember(self,item):
        self.memory.append(item)
    
    def save_models(self):
        self.network_actor.save('Actor_ddpg.h5',include_optimizer=False)
        time.sleep(1)
        self.network_critic.save('Critic_ddpg.h5',include_optimizer=False)

    def actor(self):
        last_init=tf.random_uniform_initializer(minval=-0.005,maxval=0.005)
        Inputs=tf.keras.Input(shape=(self.state_dim,))
        ac=tf.keras.layers.Dense(units=256,activation='relu')(Inputs)
        ac=tf.keras.layers.Dense(units=256,activation='relu')(ac)
        ac_output=tf.keras.layers.Dense(units=1,activation='tanh',kernel_initializer=last_init)(ac)

        ac_output=ac_output*env.action_space.high[0]
        self.actor_model=tf.keras.Model(Inputs,ac_output)
        return self.actor_model

    def critico(self):
        Inputs=tf.keras.Input(shape=(self.state_dim,))
        c=tf.keras.layers.Dense(units=16,activation='relu')(Inputs)
        c=tf.keras.layers.Dense(units=32,activation='relu')(c)
        
        action_input=tf.keras.Input(shape=(self.action_dim,))
        action_output=tf.keras.layers.Dense(units=32,activation='relu')(action_input)

        concatenate=tf.keras.layers.Concatenate()([c,action_output])
        cr=tf.keras.layers.Dense(units=256,activation='relu')(concatenate)
        cr=tf.keras.layers.Dense(units=256,activation='relu')(cr)
        c_output=tf.keras.layers.Dense(units=1)(cr)
        self.critic_model=tf.keras.Model([Inputs,action_input],c_output)
        return self.critic_model

    @tf.function
    def train_ddpg(self,state,action,reward,next_state,done):
        with tf.GradientTape() as t_critic, tf.GradientTape() as t_actor:
            target_action=self.target_actor(next_state,training=True)
            y_true=reward + self.gamma * (1-done) * self.target_critic([next_state,target_action],training=True)
            valor_critic=self.network_critic([state,action],training=True)
            loss_critic=tf.math.reduce_mean(tf.math.square(y_true - valor_critic))
            
            action_=self.network_actor(state,training=True)
            valor_critic=self.network_critic([state,action_],training=True)
            loss_actor= -tf.math.reduce_mean(valor_critic)
        
        gradient_critic=t_critic.gradient(loss_critic,self.network_critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradient_critic,self.network_critic.trainable_variables))

        gradient_actor=t_actor.gradient(loss_actor,self.network_actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradient_actor,self.network_actor.trainable_variables))
        self.update_networks()
    
    def train_model(self,batch_size):
        sample=random.sample(self.memory,batch_size)
        state_batch,action_batch,reward_batch,next_state_batch,done_batch=[],[],[],[],[]
        for s,a,r,n,d in sample:
            state_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_state_batch.append(n)
            done_batch.append(d)
        
        state_batch=tf.convert_to_tensor(state_batch)
        action_batch=tf.convert_to_tensor(action_batch)
        reward_batch=tf.cast(tf.convert_to_tensor(reward_batch),dtype=tf.float32)
        next_state_batch=tf.convert_to_tensor(next_state_batch)
        done_batch=tf.cast(tf.convert_to_tensor(done_batch),dtype=tf.float32)
        self.train_ddpg(state_batch,action_batch,reward_batch,next_state_batch,done_batch)
    
    def choose_action(self,state,noise_object):
        state=tf.convert_to_tensor(tf.expand_dims(state,0))
        action=self.network_actor(state)
        noise=noise_object()
        action = tf.squeeze(action.numpy()) + noise
        action = tf.clip_by_value(action,self.action_low,self.action_high)
        return action.numpy()

    @tf.function    
    def update_networks(self):
        for (a,b) in zip(self.target_actor.variables,self.network_actor.variables):
            a.assign(a * self.tau + b * (1 - self.tau))
        for (a,b) in zip(self.target_critic.variables,self.network_critic.variables):
            a.assign(a *  self.tau + b * (1 - self.tau))



class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * tf.sqrt(self.dt).numpy() * tf.random.normal(shape=self.mean.shape).numpy()
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = tf.zeros_like(input=self.mean).numpy()


# In[3]:


try:
    model=tf.keras.models.load_model('Actor_ddpg.h5')
    time.sleep(2)
    model_=tf.keras.models.load_model('Critic_ddpg.h5')
    load=True
    del model
    del model_
except:
    print('No se encontraron pesos')
    load=False

episode=500
env=gym.make('MountainCarContinuous-v0')
tau=0.995
gamma=0.97
std_dev = 0.50
noise=OUActionNoise(mean=tf.zeros(shape=[1]).numpy(), std_deviation=float(std_dev) * tf.ones(shape=[1]).numpy())
maxlen=15000
batch_size=32
agent=DDPG(tau=tau,gamma=gamma,max_len=maxlen,load=load)

state=env.reset()
for i in range(episode):
    action=env.action_space.sample()
    next_state,reward,done,_=env.step(action)
    agent.remember(Tupla(state,action,reward,next_state,done))
    if done:
        state=env.reset()
    else:
        state=next_state
print('Memory full...')
score=0.0
for i in range(1,episode):
    done=False
    state=env.reset()
    while not done:
        action=agent.choose_action(state,noise)
        next_state,reward,done,_=env.step(action)
        score += reward
        agent.remember(Tupla(state,action,reward,next_state,done))
        env.render()
        if done and (i % 10 == 0):
            print('Total rewards {} episode {}/{}'.format(round(score,2),i,episode))
            agent.save_models()
        agent.train_model(batch_size)
        state=next_state

env.close()

