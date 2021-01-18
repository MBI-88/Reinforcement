#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf 
import os,random 
import pygame as pg
import numpy as np
import matplotlib.pyplot as plt
from collections import deque,namedtuple 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(10)
np.random.seed(10)
plt.rcParams['figure.figsize']=[8,8]
plt.rcParams['figure.dpi']=100


# In[2]:


Tupla=namedtuple('Tuple',('state','action','reward','nextstate','done'))

class Enviroment(object):
    def __init__(self,mapa):
        self.mapa=np.array(mapa)
        self.next_state=None
        self.mapa_action={'Left':0,'Right':1,'Up':2,'Down':3}
        self.mapa_action_len=len(self.mapa_action)
        self.state_random={0:(0,0),1:(0,7),2:(4,0)}
        self.agen_row = 0
        self.agen_col = 0
        self.agen_row_cp = 0
        self.agen_col_cp = 0
        self.reward = 0.0
        self.lavel='Playing'
        self.min_reward = -0.5 * self.mapa.size
        self.total_reward = 0
        self.row_target = self.mapa.shape[0]-1
        self.col_target = self.mapa.shape[1]-1
        self.visited_cell = [(self.agen_row,self.agen_col)]
        self.done = False
        self.current_state = np.array([self.agen_row,self.agen_col])

    def step(self,action):
        self.valid_action(action)
      
        if (self.mapa[self.agen_row,self.agen_col] == 0.0): # Celda erronea
            self.reward = -0.75
            self.agen_col,self.agen_row = self.agen_col_cp,self.agen_row_cp

        elif ((self.agen_row,self.agen_col) == (self.row_target,self.col_target)): # Celda objetivo
                self.reward = 1.0
                self.done = True
                self.lavel = 'Win'
                self.visited_cell.append((self.agen_row,self.agen_col))
        else:
            if ((self.agen_row,self.agen_col) in self.visited_cell): # Celda repetida
                if self.reward == -0.80:
                    pass
                else:self.reward = -0.25
            else:
                self.visited_cell.append((self.agen_row,self.agen_col)) # Celda correcta
                self.reward = -0.04
                
        self.total_reward += self.reward
        if (self.total_reward <= self.min_reward):
            self.lavel = 'Defeat'
        self.next_state=np.array([self.agen_row,self.agen_col])
        self.agen_col_cp,self.agen_row_cp = self.agen_col,self.agen_row

        return self.next_state,self.reward,self.done,self.lavel

    def valid_action(self,action):# Acciones fuera del tablero
        action=self.mapa_action[action]
        # Casos criticos
        if (self.agen_col == 0 and action == 0):# Izquierda
            self.agen_row,self.agen_col=self.agen_row,self.agen_col
            self.reward = -0.80
            
        elif ( self.agen_col == self.col_target and action == 1):# Derecha
            self.agen_row,self.agen_col=self.agen_row,self.agen_col
            self.reward = -0.80
            
        elif (self.agen_row == 0  and action == 2):# Ascenso
            self.agen_row,self.agen_col=self.agen_row,self.agen_col
            self.reward = -0.80
            
        elif  (self.agen_row == self.row_target and action == 3):# Descenso
            self.agen_row,self.agen_col=self.agen_row,self.agen_col
            self.reward = -0.80
            
        else:
            self.reward = 0
            # Casos normales
            if (action == 0):
                self.agen_col -= 1
            elif (action == 1):
                self.agen_col += 1
            elif (action == 2):
                self.agen_row -= 1
            elif (action == 3):
                self.agen_row += 1
        
    def reset_env(self):
        random_state=random.choice(range(0,3))
        x,y=self.state_random[random_state]
        self.total_reward=0
        self.lavel='Playing'
        self.done=False
        self.agen_col=y
        self.agen_row=x
        self.agen_col_cp=y
        self.agen_row_cp=x
        self.visited_cell.clear()
        self.visited_cell.append((self.agen_row,self.agen_col))
        self.reward=0
        return self.current_state

def show_map(qmap):
    alto = 444
    ancho = 444
    screen = pg.display.set_mode((alto,ancho))
    pg.display.set_caption('Mapa')
    fondo=pg.image.load('Mapa.png').convert()
    pg.init()
    screen.blit(fondo,(0,0))
    nrows,ncols=qmap.mapa.shape
    pos=np.copy(qmap.mapa)
    cellWidth = ancho/ncols
    cellHeight = alto/nrows
    pg.draw.rect(screen,(255,100,50),((ncols-1)*cellWidth + 1,(nrows-1)*cellHeight + 1,cellWidth - 2,cellHeight - 2))
    for row,col in qmap.visited_cell:
        pg.draw.rect(screen,(100,100,100),(col*cellWidth + 1,row*cellHeight + 1,cellWidth - 2,cellHeight - 2))  

    pg.display.flip()

class Dueling_DDQN():
    def __init__(self,gamma=0.90,lr=0.01,max_len=1000,load_model=False):
        self.gamma=gamma
        self.lr=lr
        self.memory=deque(maxlen=max_len)
        self.epsilon_min=0.20
        self.epsilon_decay=0.99
        self.epsilon=1.0
        self.state_dim=env.current_state.shape[0]
        self.action_dim=env.mapa_action_len
        self.load_model=load_model
        self.counter=0

        if load_model:
            self.dqn=self.make_model()
            self.dqn_target=self.make_model()
            self.dqn.load_weights('Dueling_DDQN.h5')
            print('Model loaded')
        else:
            self.dqn=self.make_model()
            self.dqn_target=self.make_model()
        self.update_params()
        self.model_save_figure()
    
    def remember(self,item):
        self.memory.append(item)

    def update_epsilon(self):
        if self.load_model:
            self.epsilon = self.epsilon_min
        else:
            if round(self.epsilon,2) > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    
    def make_model(self):
        Inputs=tf.keras.layers.Input(shape=(self.state_dim,))
        hidden=tf.keras.layers.Dense(units=50,activation='relu')(Inputs)
        hidden=tf.keras.layers.Dense(units=50,activation='relu')(hidden)
        value=tf.keras.layers.Dense(units=1)(hidden)
        advance=tf.keras.layers.Dense(units=self.action_dim)(hidden)
        q_value=tf.keras.layers.Add()([value,(advance - tf.reduce_mean(advance,axis=1,keepdims=True))])
        model=tf.keras.Model(Inputs,q_value)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),loss='mse')
        return model
    
    def set_target(self,nexstate,reward,done):
        nexstate=tf.expand_dims(tf.convert_to_tensor(nexstate),0)
        action=tf.argmax(self.dqn.predict(nexstate)[0]).numpy()
        q_value = reward + self.gamma * (1-done) * self.dqn_target.predict(nexstate)[0][action]
        return q_value
    
    def learning(self,batch_size=32):
        sample = random.sample(self.memory,k=batch_size)
        target_batch,state_batch=[],[]
        for tran in sample:
            s,a,r,n,d=tran
            target=self.set_target(n,r,d)
            target_all=self.dqn.predict(tf.expand_dims(tf.convert_to_tensor(s),0))[0]
            target_all[a]=target
            target_batch.append(target_all)
            state_batch.append(s)

        self.update_epsilon()
        self.counter += 1
        if self.counter % 10 == 0:
            self.update_params()
        return self.dqn.train_on_batch(x=tf.convert_to_tensor(state_batch),y=tf.convert_to_tensor(target_batch))
    
    def update_params(self):
        self.dqn_target.set_weights(self.dqn.get_weights())
    
    def choose_action(self,observation):
        if np.random.rand() <= self.epsilon:
            return random.randrange(0,self.action_dim)
        observation=tf.expand_dims(tf.convert_to_tensor(observation),0)
        q_value=tf.argmax(self.dqn.predict(observation)[0]).numpy()
        return q_value
    
    def save_weights(self):
        self.dqn.save_weights('Dueling_DDQN.h5')
    
    def save_model(self):
        self.dqn.save('Dueling_DDQN_model.h5',include_optimizer=False)
        print('Modelo guardado')
        
    def model_save_figure(self):
        tf.keras.utils.plot_model(self.dqn,to_file='DDDQN_Modelo.png',dpi=100,show_shapes=True)


# In[3]:


mapa=[
    # 0  1  2  3  4  5  6  7   
    [ 1.,1.,1.,1.,1.,1.,1.,1.],#0
    [ 0.,1.,0.,0.,0.,1.,0.,1.],#1
    [ 0.,1.,1.,0.,1.,1.,0.,1.],#2
    [ 0.,1.,1.,0.,1.,0.,0.,0.],#3
    [ 1.,1.,1.,1.,1.,1.,1.,0.],#4
    [ 1.,0.,0.,0.,0.,1.,0.,0.],#5
    [ 1.,1.,1.,1.,1.,1.,0.,0.],#6
    [ 0.,0.,0.,0.,0.,1.,1.,1.],#7
]
lista_acciones={0:'Left',1:'Right',2:'Up',3:'Down'}


# In[4]:


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
   plt.savefig(fname='Dueling_DDQN.png',format='png',dpi=100)
   return plt.show()


# In[5]:


if os.path.exists('Dueling_DDQN.h5'):
    load=True
else:load=False

episode= 200 
gamma=0.97
learning_rate=0.0005
maxlen=16000
batch=32
env=Enviroment(mapa)
agent=Dueling_DDQN(lr=learning_rate,max_len=maxlen,gamma=gamma,load_model=load)
state=env.reset_env()
for i  in range(maxlen):
    action=agent.choose_action(state)
    accion=lista_acciones[action]
    state_,reward,done,level=env.step(accion)
    agent.remember(Tupla(state,action,reward,state_,done))
    if level != 'Playing':
        state=env.reset_env()
    else:
        state = state_
print('Memory Full..')

lista_rewards,lista_losses=[],[]
for i in range(1,episode):
    state=env.reset_env()
    score = 0
    while True:
        action=agent.choose_action(state)
        accion=lista_acciones[action]
        state_,reward,done,level=env.step(accion)
        agent.remember(Tupla(state,action,reward,state_,done))
        score += reward
        show_map(env)
        if level != 'Playing':
            if i % 10 == 0:
                print('Reward: {} Episode: {}/{} Estado'.format(round(score,2),i,episode,level))
                agent.save_weights()
            break
        state = state_
        losse = agent.learning(batch_size=batch)
    lista_rewards.append(score)
    lista_losses.append(losse)
pg.quit()


# In[6]:


plot_history(lista_losses,lista_rewards,episode)


# In[7]:


agent.save_model()


# In[10]:


def test_model():
    try:
        modelo=tf.keras.models.load_model('Dueling_DDQN_model.h5')
        env=Enviroment(mapa)
        print('Modelo cargado')
    except OSError :
        return print('No se pudo cargar el modelo')

    lista_scores=[]
    for i in range(1,20):
        score=0
        state=env.reset_env()
        state=tf.expand_dims(tf.convert_to_tensor(state),0)
        while True:
            action=tf.argmax(modelo.predict(state)[0]).numpy()
            accion=lista_acciones[action]
            state_,reward,done,level=env.step(accion)
            score += reward
            show_map(env)
            if level != 'Playing':
                print('Episodio -> {}/{}  Premio -> {} '.format(i,50,round(score,2)))
                break
                
            state = tf.expand_dims(tf.convert_to_tensor(state_),0)
        lista_scores.append(score)
    pg.quit()
    array_score=np.array(lista_scores)
    promedio = np.sum(array_score)/20

    if promedio >= -1:
        print('Valor alcanzado {} (Victoria)'.format(promedio))
    else:
        print('Valor alcanzado {} (Perdida)'.format(promedio))


# In[11]:


test_model()


# ***Nota: El modelo tardo comenzo con 500 episodios de entrenamineto , fue entrenado en varios lapsos de tiempo***
