# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 14:45:24 2021

@author: MBI
"""

import tensorflow as tf 
import gym,random,os
from collections import deque,namedtuple
import numpy as np 
import matplotlib.pyplot as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(1)
np.random.seed(1)


# %%
env=gym.make('MountainCar-v0')
env.reward_range = (-1,1)
env.action_space.n,env.observation_space.shape,env.reward_range
print(env.observation_space,' ',env.action_space.n,' ',env.reward_range)
# %%
Tuple=namedtuple('Transition',('state','action','nextstate','reward','done'))

class DDQN():
    def __init__(self,input_dim,output_dim,lr=0.001,gamma=0.99,max_len=1000,epsi=0.10,load=False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = epsi
        self.epsilon_decay = 0.999
        self.load = load
        self.memory = deque(maxlen=max_len)
        self.counter_update = 0

        if self.load:
            self.ddqn = self.make_model()
            self.ddqn_target = self.make_model()
            try:
                self.ddqn.load_weights('C:/Users/MBI/Documents/Python_Scripts/Practicas_AI/MountainCar/Pesos_ddqn.h5')
                print('[+] Pesos cargados...')
            except:
                print('[-] No se cargaron pesos')
            
        else:
            self.ddqn = self.make_model()
            self.ddqn_target = self.make_model()
            tf.keras.utils.plot_model(self.ddqn,to_file = 'C:/Users/MBI/Documents/Python_Scripts/Practicas_AI/MountainCar/Estuctura_Modelo.png',show_layer_names=True,show_shapes=True,dpi=100)
        self.set_weights()
        
            
    def make_model(self):
        inputs = tf.keras.Input(shape=(self.input_dim,))
        Capa = tf.keras.layers.Dense(units=60,activation='relu',name='Capa_1')(inputs)
        Capa = tf.keras.layers.Dense(units=50,activation='relu',name='Capa_2')(Capa)
        Q = tf.keras.layers.Dense(units=self.output_dim,name="Q_funtion")(Capa)
        modelo = tf.keras.Model(inputs,Q)
        modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),loss='mse')
        return modelo
    
    def add(self,item):
        self.memory.append(item)
    
    def set_weights(self):
        self.ddqn_target.set_weights(self.ddqn.get_weights())

    def save_weights(self):
        self.ddqn.save_weights('C:/Users/MBI/Documents/Python_Scripts/Practicas_AI/MountainCar/Pesos_ddqn.h5')

    def save_model(self):
        self.ddqn.save('C:/Users/MBI/Documents/Python_Scripts/Practicas_AI/MountainCar/Modelo_DDQN.h5',include_optimizer=False)
        print('[+] Modelo guardado...')

    def get_qvalue(self,nextstate,reward,done):
        nextstate = tf.expand_dims(tf.convert_to_tensor(nextstate),0)
        action = tf.argmax(self.ddqn.predict(nextstate)[0]).numpy()
        q_value = reward + self.gamma * (1-done) * self.ddqn_target.predict(nextstate)[0][action]
        return q_value

    def _learning(self,batch_size=32):
        sample = random.sample(self.memory,batch_size)
        target_batch,state_batch=[],[]
        for trans in sample:
            s,a,n,r,d = trans
            target = self.get_qvalue(n,r,d)
            prediction = self.ddqn.predict(tf.expand_dims(tf.convert_to_tensor(s),0))[0]
            prediction[a] = target
            target_batch.append(prediction)
            state_batch.append(s)
          
        self.counter_update += 1
        
        if self.counter_update % 10 == 0:
            self.set_weights()

        self.update_epsilon()
        return self.ddqn.train_on_batch(x=tf.convert_to_tensor(state_batch),y=tf.convert_to_tensor(target_batch))
    
    def update_epsilon(self):
        if self.load:
            self.epsilon = self.epsilon_min
        else:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def choose_action(self,observation):
        if np.random.rand() <= self.epsilon:
            return random.randrange(0,self.output_dim)
        else:
            observation=tf.expand_dims(tf.convert_to_tensor(observation),0)
            action=tf.argmax(self.ddqn.predict(observation)[0]).numpy()
            return action


#%%
def plot_history(losses,rewards,epochs):
   fig,(axis_1,axis_2)=plt.subplots(1,2,sharey=False,sharex=False)
   axis_1.plot((range(min(epochs,len(losses)))),losses,color='b')
   axis_1.set_title('Perdidas por epocas',color='r',size=12)
   axis_1.set_xlabel('Epocas',color='r',size=10)
   axis_1.set_ylabel('Perdidas',color='r',size=10)
   axis_2.plot((range(min(epochs,len(rewards)))),rewards,color='r')
   axis_2.set_title('Premios por epocas',color='r',size=12)
   axis_2.set_xlabel('Epocas',color='r',size=10)
   axis_2.set_ylabel('Premios',color='r',size=10)
   plt.savefig(fname='C:/Users/MBI/Documents/Python_Scripts/Practicas_AI/MountainCar/DDQN.png',format='png',dpi=100)
   return plt.show()


#%%
if os.path.exists('C:/Users/MBI/Documents/Python_Scripts/Practicas_AI/MountainCar/Pesos_ddqn.h5'):
    load=True
else: load = False

inputdim=env.observation_space.shape[0]
outputdim=env.action_space.n
episodios = 500
learning_rate = 1e-3
memory_len = 30000
batch = 32
gamma = 0.99
epsilon_min = 0.10

agente = DDQN(lr=learning_rate,max_len=memory_len,load=load,input_dim=inputdim,output_dim=outputdim,gamma=gamma,epsi=epsilon_min)
state = env.reset()

for i in range(memory_len):
    action = env.action_space.sample()
    nstate,reward,done,info = env.step(action)
    agente.add(Tuple(state,action,nstate,reward,done))
    if done:
        state = env.reset()
    else:
        state = nstate
print('[*] Memoria llena...')

losses,rewards = [],[]
for i in range(1,episodios):
    state = env.reset()
    done = False
    score = 0
    loss = 0
    while not done:
        action = agente.choose_action(state)
        nstate,reward,done,info = env.step(action)
        agente.add(Tuple(state,action,nstate,reward,done))
        env.render()
        score += reward
        if done:
            print('[*] Episodio -> {}/{}  Premio -> {} '.format(i,episodios,score))
            agente.save_weights()

        state = nstate
        loss += agente._learning(batch)
    losses.append(loss)
    rewards.append(score)
env.close()


#%%
plot_history(losses,rewards,episodios)

# %%
agente.save_model()

#%%
def test_model():
    try:
        modelo = tf.keras.models.load_model('C:/Users/MBI/Documents/Python_Scripts/Practicas_AI/MountainCar/Modelo_DDQN.h5')
        env = gym.make('MountainCar-v0')
        print('[+] Modelo cargado')
    except OSError :
        return print('[-] No se pudo cargar el modelo')
    lista_scores = []
    for i in range(1,100):
        score = 0
        state = env.reset()
        state = tf.expand_dims(tf.convert_to_tensor(state),0)
        done = False
        while not done:
            action = tf.argmax(modelo.predict(state)[0]).numpy()
            nstate,reward,done,info = env.step(action)
            env.render()
            score += reward
            if done:
                print('[*] Episodio -> {}/{}  Premio -> {} '.format(i,100,score))
                
            state = tf.expand_dims(tf.convert_to_tensor(nstate),0)
        lista_scores.append(score)
    env.close()
    
    promedio = sum(lista_scores)/100

    if promedio >= -110:
        print('[+] Valor alcanzado {} (Victoria)'.format(promedio))
    else:
        print('[-] Valor alcanzado {} (Perdida)'.format(promedio))


# %%
print('[*] Test Model..')
test_model()
#%%
env.close()
#%%
