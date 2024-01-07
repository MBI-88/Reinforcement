#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import tensorflow as tf 
import tensorflow_probability as tfp 
import gym ,math
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(1)
np.random.seed(1)


# In[2]:


def softplusk(x):
    return tf.keras.activations.softplus(x=x) + 1e-10

class Piliticy_Garadient():
    def __init__(self,env):
        self.env=env
        self.beta=0.0
        self.memory=[]
        self.epsilon=1.0
        self.epsilon_decay=0.999
        self.epsilon_min=0.10
        self.state=env.reset()
        self.state_dim=env.observation_space.shape[0]
        self.state=np.reshape(self.state,[1,self.state.shape[0]])
        self.build_autoencoder()

    def reset_memory(self):
        self.memory=[]
    def remember(self,item):
        self.memory.append(item)
    
    def action(self,arg):
        mean,std=arg
        dist=tfp.distributions.Normal(loc=mean,scale=std)
        action_dist=dist.sample(1)
        action=tf.clip_by_value(action_dist,self.env.action_space.low[0],self.env.action_space.high[0])
        return action
    
    def logp(self,arg):
        mean,std,action=arg
        dist=tfp.distributions.Normal(loc=mean,scale=std)
        lgp=dist.log_prob(action)
        return lgp
    
    def entropy(self,arg):
        mean,std=arg
        dist=tfp.distributions.Normal(loc=mean,scale=std)
        entropy=dist.entropy()
        return entropy
    
    def build_autoencoder(self):
        # Encoder
        inputs=tf.keras.Input(shape=(self.state_dim,))
        feature_size=64
        X=tf.keras.layers.Dense(units=512,activation='relu')(inputs)
        X=tf.keras.layers.Dense(units=256,activation='relu')(X)
        X=tf.keras.layers.Dense(units=128,activation='relu')(X)
        output_enco=tf.keras.layers.Dense(units=feature_size)(X)
        self.encoder=tf.keras.Model(inputs,output_enco)

        # Decoder
        feature_inputs=tf.keras.Input(shape=(feature_size,))
        Y=tf.keras.layers.Dense(units=128,activation='relu')(feature_inputs)
        Y=tf.keras.layers.Dense(units=256,activation='relu')(Y)
        Y=tf.keras.layers.Dense(units=512,activation='relu')(Y)
        output_deco=tf.keras.layers.Dense(units=self.state_dim)(Y)
        self.decoder=tf.keras.Model(feature_inputs,output_deco)

        # Autoencoder
        self.autoencoder=tf.keras.Model(inputs,self.decoder(self.encoder(inputs)))
        self.autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-3),loss='mse')

    def train_autoencoder(self,x_train,y_test):
        self.autoencoder.fit(x_train,x_train,validation_data=(y_test,y_test),batch_size=32,epochs=10)
    
    def build_model(self):
        Inputs=tf.keras.Input(shape=(self.state_dim,))
        self.encoder.trainable=False
        x=self.encoder(Inputs)
        mean=tf.keras.layers.Dense(units=1,kernel_initializer='zero')(x)
        std=tf.keras.layers.Dense(units=1,kernel_initializer='zero')(x)
        std=tf.keras.layers.Activation('softplusk')(std)
        action=tf.keras.layers.Lambda(self.action,output_shape=(1,))([mean,std])
        logp=tf.keras.layers.Lambda(self.logp,output_shape=(1,))([mean,std,action])
        entropy=tf.keras.layers.Lambda(self.entropy,output_shape=(1,))([mean,std])
        valor=tf.keras.layers.Dense(units=1,kernel_initializer='zero')(x)

        # Modelo
        self.logp_model=tf.keras.Model(Inputs,logp)
        self.entropy_model=tf.keras.Model(Inputs,entropy)
        self.actor=tf.keras.Model(Inputs,action)
        self.valor=tf.keras.Model(Inputs,valor)

        # losses
        loss=self.logp_loss(self.get_entropy(self.state),beta=self.beta)
        optimizer_actor=tf.keras.optimizers.RMSprop(learning_rate=3e-3)
        self.logp_model.compile(optimizer=optimizer_actor,loss=loss)
        optimizer_valor=tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.valor.compile(optimizer=optimizer_valor,loss=self.loss_value)


    def logp_loss(self,entropy,beta=0.0):
        def loss(y_true,y_pred):
           return -tf.keras.backend.mean((y_true*y_pred)+(beta*entropy),axis=1)
        return loss
    
    def get_entropy(self,state):
        entropy=self.entropy_model.predict(state)
        return entropy[0]

    def choose_action(self,state):
        action=self.actor.predict(state)
        return action[0]

    def loss_value(self,y_true,y_pred):
        return -tf.keras.backend.mean(y_true * y_pred,axis=-1)

    def saved_weight(self):
        self.actor.save_weights('Actor.h5')
        self.encoder.save_weights('Encoder_Weight.h5')
        self.valor.save_weights('Critic.h5')

    def load_weight(self):
        self.actor.load_weights('Actor.h5')
        self.valor.load_weights('Critic.h5')
    
    def load_encode_weight(self):
        self.encoder.load_weights('Encoder_Weight.h5')
        
    def save_model(self):
        self.actor.save('Model.h5')

class Actor_Critico(Piliticy_Garadient):
    def __init__(self,env):
        super().__init__(env=env)

    def train(self,item,gamma=1.0):
        [step, state, next_state, reward, done] = item
        self.state = state
        discount_factory= gamma**step
        delta = reward - self.valor.predict(state)[0]

        if not done:
            next_value=self.valor.predict(next_state)[0]
            delta += gamma*next_value

        discount_delta = delta * discount_factory
        discount_delta = np.reshape(discount_delta,[-1,1])

        history_logp=self.logp_model.fit(np.array(state),discount_delta,batch_size=1,epochs=1,verbose=0)
        history_valor=self.valor.fit(np.array(state),discount_delta,batch_size=1,epochs=1,verbose=0)


# In[3]:


episode=500
env=gym.make('MountainCarContinuous-v0')
agent=Actor_Critico(env=env)
tf.keras.utils.get_custom_objects().update({'softplusk':tf.keras.layers.Activation(softplusk)})
try:
   agent.load_encode_weight()
except:
    print('No se encontro pesos / Muestreando ambiente...')
    x_train=[env.observation_space.sample() for i in range(200000)]
    y_test=[env.observation_space.sample() for i in range(200000)]
    x_train=np.array(x_train)
    y_test=np.array(y_test)
    agent.train_autoencoder(x_train,y_test)
    
try:
    agent.load_weight()
except:
    print('No se encontro pesos')
agent.build_model()
score=0.0

for i in range(1,episode):
    state=env.reset()
    state=np.reshape(state,[1,state.shape[0]])
    done=False
    step=0
    agent.reset_memory()
    while not done:
        action=agent.choose_action(state)
        next_state,reward,done,_=env.step(action)
        next_state=np.reshape(next_state,[1,next_state.shape[0]])
        item=[step, state, next_state, reward, done]
        agent.remember(item)
        score += reward
        state=next_state
        step += 1
        env.render()
        agent.train(item,gamma=0.99)
        if done:
            if i % 10==0:
                print('Episode: {}/{} Total Rewarsd: {}'.format(i,episode,score)) 
                agent.saved_weight()
env.close()


# In[ ]:




