#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf 
import gym,random,os
from collections import deque,namedtuple 
import tensorflow_probability as tfp
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(1)


# In[2]:


Tupla=namedtuple('Tupla',('state','action','reward','next_state','done'))

class SAC():
    def __init__(self,tau=1,gamma=0.95,max_len=1000,load=False):
        self.tau=tau
        self.gamma=gamma
        self.memory=deque(maxlen=max_len)
        self.load=load
        self.max_action=env.action_space.high
        self.sigma=1.0
        self.epsilon_min=0.10
        self.logprob_epsilon=1e-6
        self.epsilon_decay=0.95
        self.action_low=env.action_space.low[0]
        self.action_high=env.action_space.high[0]
        self.action_dim=env.action_space.shape[0]
        self.state_dim=env.observation_space.shape[0]
        self.optimizer_polit=tf.keras.optimizers.Adam(learning_rate=0.001)
        self.optimizer_q1=tf.keras.optimizers.Adam(learning_rate=0.002)
        self.optimizer_q2=tf.keras.optimizers.Adam(learning_rate=0.002)
        self.autoencode()

        if not self.load:
            self.network_p=self.policitie()
            self.q1=self.function_q()
            self.q2=self.function_q()
            self.q1_target=self.function_q()
            self.q2_target=self.function_q()
        else:
            self.network_p=self.policitie()
            self.q1=self.function_q()
            self.q2=self.function_q()
            self.network_p.load_weights('SAC_politicie.h5')
            self.q1.load_weights('SAC_q1.h5')
            self.q2.load_weights('SAC_q2.h5')
            self.q1_target=self.function_q()
            self.q2_target=self.function_q()

        self.update_params()


    def autoencode(self):
        Inputs_e=tf.keras.Input(shape=(self.state_dim,))
        X=tf.keras.layers.Dense(units=512,activation='relu')(Inputs_e)
        X=tf.keras.layers.Dropout(rate=0.30)(X)
        X=tf.keras.layers.Dense(units=256,activation='relu')(X)
        X=tf.keras.layers.Dropout(rate=0.30)(X)
        X=tf.keras.layers.Dense(units=128,activation='relu')(X)
        outputs_e=tf.keras.layers.Dense(units=64)(X)
        self.encode=tf.keras.Model(Inputs_e,outputs_e)
        
        Inputs_d=tf.keras.Input(shape=(64,))
        Y=tf.keras.layers.Dense(units=128,activation='relu')(Inputs_d)
        Y=tf.keras.layers.Dropout(rate=0.30)(Y)
        Y=tf.keras.layers.Dense(units=256,activation='relu')(Y)
        Y=tf.keras.layers.Dropout(rate=0.30)(Y)
        Y=tf.keras.layers.Dense(units=512,activation='relu')(Y)
        outputs_d=tf.keras.layers.Dense(units=self.state_dim)(Y)
        self.decode=tf.keras.Model(Inputs_d,outputs_d)

        self.autoencoder=tf.keras.Model(Inputs_e,self.decode(self.encode(Inputs_e)))
        self.autoencoder.compile(optimizer='adam',loss='mse')
        
    def train_autoencoder(self,x_train,y_train):
        self.autoencoder.fit(x_train,x_train,validation_data=(y_train,y_train),batch_size=32,epochs=10)
    
    def dist_fuction(self,item):
        mean,stdv = item
        stdv = tf.clip_by_value(stdv,-20,2)
        dist = tfp.distributions.Normal(mean,tf.exp(stdv))
        action=dist.sample()
        squashed_action=tf.tanh(action) 
        log_p = dist.log_prob(action) - tf.math.log(1.0 - tf.pow(squashed_action,2) + self.logprob_epsilon)
        log_p = tf.reduce_sum(log_p,axis=-1,keepdims=True)
        return squashed_action,log_p

    def policitie(self):
        Inputs=tf.keras.Input(shape=(self.state_dim,))
        self.encode.trainable=False
        X=self.encode(Inputs)
        #X=tf.keras.layers.Dense(units=256,activation='relu')(Inputs)
        #X=tf.keras.layers.Dense(units=256,activation='relu')(X)
        mean=tf.keras.layers.Dense(units=1,kernel_initializer=tf.random_uniform_initializer(minval=-0.003,maxval=0.003),bias_initializer=tf.random_uniform_initializer(minval=-0.003,maxval=0.003))(X)
        stdv=tf.keras.layers.Dense(units=1,kernel_initializer=tf.random_uniform_initializer(minval=-0.003,maxval=0.003),bias_initializer=tf.random_uniform_initializer(minval=-0.003,maxval=0.003))(X)
        action,log_p=tf.keras.layers.Lambda(self.dist_fuction,output_shape=(2,))([mean,stdv])
        self.model_policitie=tf.keras.Model(Inputs,[action,log_p])
        return self.model_policitie
  
    def function_q(self):
        Inputs=tf.keras.Input(shape=self.state_dim,)
        aq=tf.keras.layers.Dense(units=16,activation='relu')(Inputs)
        aq=tf.keras.layers.Dense(units=32,activation='relu')(aq)

        action_p=tf.keras.Input(shape=self.action_dim,)
        aaq=tf.keras.layers.Dense(units=32,activation='relu')(action_p)

        concat=tf.keras.layers.Concatenate()([aq,aaq])
        output_q=tf.keras.layers.Dense(units=256,activation='relu')(concat)
        output_q=tf.keras.layers.Dense(units=256,activation='relu')(output_q)
        output_q=tf.keras.layers.Dense(units=1,kernel_initializer=tf.random_uniform_initializer(minval=-0.003,maxval=0.003),bias_initializer=tf.random_uniform_initializer(minval=-0.003,maxval=0.003))(output_q)
        self.model_q=tf.keras.Model([Inputs,action_p],output_q)
        return self.model_q

    @tf.function
    def update_params(self):
        for (a,b) in zip(self.q1_target.variables,self.q1.variables):
            a.assign(a * self.tau + b * (1-self.tau))
        for (a,b) in zip(self.q2_target.variables,self.q2.variables):
            a.assign(a * self.tau + b * (1-self.tau))
    
    def remember(self,item):
        self.memory.append(item)

    def update_epsilon(self):
        if self.sigma >= self.epsilon_min:
            self.sigma *= self.epsilon_decay
    
    def sample_memory(self,batch_size):
        sample=random.sample(self.memory,batch_size)
        state_b,action_b,reward_b,nextstate_b,done_b=[],[],[],[],[]
        for s,a,r,n,d in sample:
            state_b.append(s)
            action_b.append(a)
            reward_b.append(r)
            nextstate_b.append(n)
            done_b.append(d)
        
        state_b=tf.convert_to_tensor(state_b)
        action_b=tf.convert_to_tensor(action_b)
        reward_b=tf.cast(tf.convert_to_tensor(reward_b),dtype=tf.float32)
        nextstate_b=tf.convert_to_tensor(nextstate_b)
        done_b=tf.cast(tf.convert_to_tensor(done_b),dtype=tf.float32)
        self.train_sac(state_b,action_b,reward_b,nextstate_b,done_b)
        self.update_params()
        
    @tf.function
    def train_sac(self,state,action,reward,next_state,done):
        with tf.GradientTape() as t_q1, tf.GradientTape() as t_q2, tf.GradientTape() as t_policities:
            action_,logp=self.network_p(next_state,training=True)
            y_true = reward + self.gamma * (1-done) * (tf.minimum(self.q1_target([next_state,action_]),self.q2_target([next_state,action_])) - self.sigma * logp)
            q1_pred=self.q1([state,action],training=True)
            q2_pred=self.q2([state,action],training=True)
            q1_loss=tf.keras.losses.MSE(y_true,q1_pred)
            q2_loss=tf.keras.losses.MSE(y_true,q2_pred)

            actions,logp_=self.network_p(state,training=True)
            valor_q1=self.q1([state,actions],training=True)
            valor_q2=self.q2([state,actions],training=True)
            #avd=tf.stop_gradient(logp_- tf.minimum(valor_q1,valor_q2))
            network_loss=tf.reduce_mean(logp_ - tf.minimum(valor_q1,valor_q2))
        
        gradient_tq1=t_q1.gradient(q1_loss,self.q1.trainable_variables)
        self.optimizer_q1.apply_gradients(zip(gradient_tq1,self.q1.trainable_variables))
        gradient_tq2=t_q2.gradient(q2_loss,self.q2.trainable_variables)
        self.optimizer_q2.apply_gradients(zip(gradient_tq2,self.q2.trainable_variables))
        gradient_p=t_policities.gradient(network_loss,self.network_p.trainable_variables)
        self.optimizer_polit.apply_gradients(zip(gradient_p,self.network_p.trainable_variables))
    
    def choose_action(self,observation):
        state=tf.expand_dims(tf.convert_to_tensor(observation),0)
        action,_=self.network_p(state)
        action=tf.reshape(action,shape=[-1])
        return action.numpy()
    
    def load_encoder(self):
        self.encode.load_weights('Encode_weights.h5')
    
    def save_models(self):
        self.q1.save_weights('SAC_q1.h5')
        self.q2.save_weights('SAC_q2.h5')
        self.network_p.save_weights('SAC_politicie.h5')
        self.encode.save_weights('Encode_weights.h5')
 


# In[3]:



if (os.path.exists('SAC_politicie.h5') and os.path.exists('SAC_q1.h5') and  os.path.exists('SAC_q2.h5')): load=True
else:load=False

env=gym.make('MountainCarContinuous-v0')
tau=0.995
gamma=0.97
maxlen=15000
episode=500
batch_size=32
agent=SAC(tau=tau,gamma=gamma,load=load,max_len=maxlen)

try:
    agent.load_encoder()
except:
    print('No se encontro pesos / Muestreando ambiente...')
    x_train=[env.observation_space.sample() for i in range(200000)]
    y_test=[env.observation_space.sample() for i in range(200000)]
    x_train=tf.convert_to_tensor(x_train)
    y_test=tf.convert_to_tensor(y_test)
    agent.train_autoencoder(x_train,y_test)

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
            agent.update_epsilon()
        agent.sample_memory(batch_size)
        state=next_state

env.close()

