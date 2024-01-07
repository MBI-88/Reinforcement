#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf 
import gym,random,time
from collections import deque,namedtuple 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(1)


# In[2]:


Tupla=namedtuple('Tupla',('state','action','reward','next_state','done'))

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

class TD3():
    def __init__(self,tau=0.005,noise_train=None,noise_action=None,gamma=0.95,load=False,max_len=1000):
        self.tau=tau
        self.noise_train=noise_train
        self.noise_action=noise_action
        self.lr_critic=0.002
        self.lr_actor=0.001
        self.gamma=gamma
        self.counter_actor=tf.constant(2)
        self.update_actor=tf.constant(0)
        self.memory=deque(maxlen=max_len)
        self.optimizer_actor=tf.keras.optimizers.Adam(learning_rate=0.001)
        self.optimizer_critic_1=tf.keras.optimizers.Adam(learning_rate=0.002)
        self.optimizer_critic_2=tf.keras.optimizers.Adam(learning_rate=0.002)
        self.action_dim=env.action_space.shape[0]
        self.state_dim=env.observation_space.shape[0]
        self.action_low=env.action_space.low[0]
        self.action_high=env.action_space.high[0]
        self.load=load
        if  not self.load:
            self.network_actor=self.actor()
            self.network_critic_1=self.critico()
            self.network_critic_2=self.critico()
            self.target_actor=self.actor()
            self.target_critic_1=self.critico()
            self.target_critic_2=self.critico()
        else:
            self.network_actor=tf.keras.models.load_model('Actor_TD3.h5')
            self.target_actor=self.actor()
            time.sleep(1)
            self.network_critic_1=tf.keras.models.load_model('Critic_TD3_1.h5')
            self.target_critic_1=self.critico()
            time.sleep(1)
            self.network_critic_2=tf.keras.models.load_model('Critic_TD3_2.h5')
            self.target_critic_2=self.critico()
        self.update_params()
    
    def remember(self,item):
        self.memory.append(item)
    
    def save_models(self):
        self.network_actor.save('Actor_TD3.h5',include_optimizer=False)
        time.sleep(0.5)
        self.network_critic_1.save('Critic_TD3_1.h5',include_optimizer=False)
        time.sleep(0.5)
        self.network_critic_2.save('Critic_TD3_2.h5',include_optimizer=False)
    
    def actor(self):
        last_init=tf.random_uniform_initializer(minval=-0.003,maxval=0.003)
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
    def update_params(self):
        for (a,b) in zip(self.target_actor.variables,self.network_actor.variables):
            a.assign(a * self.tau + b * (1-self.tau))
        for (a,b) in zip(self.target_critic_1.variables,self.network_critic_1.variables):
            a.assign(a * self.tau + b * (1-self.tau))
        for (a,b) in zip(self.target_critic_2.variables,self.network_critic_2.variables):
            a.assign(a * self.tau + b * (1-self.tau))
    
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
        noise=tf.convert_to_tensor(tf.cast(self.noise_train(),dtype=tf.float32))
        self.train_Q(state_batch,action_batch,reward_batch,next_state_batch,done_batch,noise)
        self.update_actor += 1

        if self.update_actor % self.counter_actor != 0:
            pass
        else:
            self.train_P(state_batch)
    
    @tf.function  
    def train_Q(self,state,action,reward,next_state,done,noise):
        with  tf.GradientTape() as t_critic_1, tf.GradientTape() as t_critic_2:
            target_action = tf.clip_by_value(self.target_actor(next_state,training=True) + tf.clip_by_value(noise,-0.5,0.5),self.action_low,self.action_high)
            y_true = reward + self.gamma * (1-done) * tf.minimum(self.target_critic_1([next_state,target_action],training=True),self.target_critic_2([next_state,target_action],training=True))
            valor_critic_1 = self.network_critic_1([state,action],training=True)
            valor_critic_2 = self.network_critic_2([state,action],training=True)
            loss_critic_1 = tf.keras.losses.MSE(y_true,valor_critic_1)
            loss_critic_2 = tf.keras.losses.MSE(y_true,valor_critic_2)

        gradient_critic_1 = t_critic_1.gradient(loss_critic_1,self.network_critic_1.trainable_variables)
        self.optimizer_critic_1.apply_gradients(zip(gradient_critic_1,self.network_critic_1.trainable_variables))
        gradient_critic_2 = t_critic_2.gradient(loss_critic_2,self.network_critic_2.trainable_variables)
        self.optimizer_critic_2.apply_gradients(zip(gradient_critic_2,self.network_critic_2.trainable_variables))

    @tf.function
    def train_P(self,state):
        with tf.GradientTape() as t_actor:
            action_  = self.network_actor(state,training=True)
            valor_critic = self.network_critic_1([state,action_],training=True)
            loss_actor = -tf.math.reduce_mean(valor_critic)
            
        gradient_actor = t_actor.gradient(loss_actor,self.network_actor.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(gradient_actor,self.network_actor.trainable_variables))
        self.update_params()
    
    def choose_action(self,state):
        state=tf.convert_to_tensor(tf.expand_dims(state,0))
        action=self.network_actor(state)
        action=tf.squeeze(action.numpy()) + self.noise_action()
        action=tf.clip_by_value(action,self.action_low,self.action_high)
        return action.numpy()


# In[3]:


try:
    model=tf.keras.models.load_model('Actor_TD3.h5')
    time.sleep(1)
    model_=tf.keras.models.load_model('Critic_TD3_1.h5')
    time.sleep(1)
    model__=tf.keras.models.load_model('Critic_TD3_2.h5')
    load=True
    del model
    del model_
    del model__
except:
    print('No se encontraron pesos')
    load=False

episode=500
gamma=0.98
tau=0.995
batch_size=32
stddv_t=1.0
stddv_a=0.5
maxlen=15000
env=gym.make('MountainCarContinuous-v0')
noise_t=OUActionNoise(mean=tf.zeros(shape=[1]).numpy(), std_deviation=float(stddv_t) * tf.ones(shape=[1]).numpy())
noise_a=OUActionNoise(mean=tf.zeros(shape=[1]).numpy(), std_deviation=float(stddv_a) * tf.ones(shape=[1]).numpy())
agent=TD3(tau=tau,noise_train=noise_t,noise_action=noise_a,gamma=gamma,load=load,max_len=maxlen)


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
        action=agent.choose_action(state)
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

