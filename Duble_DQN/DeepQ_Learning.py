#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf 
import numpy as np 
from collections import deque,namedtuple 
import random,time
import pygame as pg 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(10)
np.random.seed(10)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Ambiente 
class Enviroment(object):
    def __init__(self,mapa):
        self.mapa=np.array(mapa)
        self.next_state=None
        self.mapa_action={'Left':0,'Right':1,'Up':2,'Down':3}
        self.mapa_action_len=len(self.mapa_action)
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
        self.total_reward=0
        self.lavel='Playing'
        self.done=False
        self.agen_col=0
        self.agen_row=0
        self.agen_col_cp=0
        self.agen_row_cp=0
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
    nrows,ncols=env.mapa.shape
    pos=np.copy(env.mapa)
    cellWidth = ancho/ncols
    cellHeight = alto/nrows
    pg.draw.rect(screen,(255,100,50),((ncols-1)*cellWidth + 1,(nrows-1)*cellHeight + 1,cellWidth - 2,cellHeight - 2))
    for row,col in env.visited_cell:
        pg.draw.rect(screen,(100,100,100),(col*cellWidth + 1,row*cellHeight + 1,cellWidth - 2,cellHeight - 2))  

    pg.display.flip()


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


# Agnete 
Transition=namedtuple('Transition',('current_state','reward','action','next_state','done'))

class DDQN(object):
    def __init__(self,max_memory=100,lr=0.001,epsilon_greedy=1.0,epsilon_min=0.1,gamma=0.999,epsilon_decay=0.99,model_load=None):
        self.max_memory=max_memory
        self.lr=lr 
        self.epsilon=epsilon_greedy
        self.epsilon_decay=epsilon_decay
        self.epsilon_min=epsilon_min
        self.gamma=gamma
        self.input_state=env.current_state.shape[0]
        self.output_action=env.mapa_action_len
        self.memory=deque(maxlen=self.max_memory)
        self.counter_replay=0
        self.model=tf.keras.Sequential([
                # Capa de entrada
                tf.keras.layers.Input(shape=(self.input_state,)),
                # Capas ocultas
                tf.keras.layers.Dense(units=env.mapa.size,activation='relu'),
                tf.keras.layers.Dropout(rate=0.05),
                tf.keras.layers.Dense(units=env.mapa.size,activation='relu'),
                tf.keras.layers.Dropout(rate=0.050),
                # Capa de salida
                tf.keras.layers.Dense(units=self.output_action,activation='softmax')
            ])

        if model_load == None:
            self.q_network=self.model
            self.q_network.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                    loss='mse')
            self.q_target_model=self.model
        else:
            self.q_network=model_load
            self.q_target_model=self.model
            self.copy_pesos()
            print('Modelo cargado')
            time.sleep(1.2)
      
    def save_model(self):
        self.q_network.save('DDQN.h5')

    def copy_pesos(self):
        self.q_target_model.set_weights(self.q_network.get_weights())

    def get_qtarget(self,next_state,reward):
        action=np.argmax(self.q_network.predict(next_state)[0])
        q_value=self.q_target_model.predict(next_state)[0][action]
        q_value *= self.gamma
        q_value += reward
        return q_value

    def remember(self,transition):
        self.memory.append(transition)
    
    def choose_action(self,state):
        if np.random.rand() <= self.epsilon: # Exploracion
            return random.randrange(0,self.output_action)
        Q_value=self.q_network.predict(state)[0] # Explotacion
        return np.argmax(Q_value)
    
    def _learning(self,batch_memory):
        batch_input,batch_target = [],[]
        for transition in batch_memory:
            currrent_state,reward,action,next_state,done = transition
            if done:
                target = reward
            else:
                target = self.get_qtarget(next_state,reward)

            target_all=self.model.predict(currrent_state)[0]
            target_all[action]=target
            batch_input.append(currrent_state.flatten())
            batch_target.append(target_all)
            self.reajust_epsilon()
        return self.q_network.train_on_batch(x=np.array(batch_input),y=np.array(batch_target))
    
    def reajust_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def replay(self,batch_memory):
        sample=random.sample(self.memory,batch_memory)
        history=self._learning(sample)
        if self.counter_replay % 10 == 0:
            self.copy_pesos()
        self.counter_replay += 1


# In[5]:


try:
    load=tf.keras.models.load_model('DDQN.h5')
except:
    print('Load vacio')
    load=None


# In[6]:


# Entrenamiento 
episodio=5000
batch_size=32
memory_len=12000
learning_rate=0.003
gamma = 0.9
epsilon_decay = 0.995
epsilon_min = 0.30

# Instanciando Agente y Ambiente
env=Enviroment(mapa=mapa)
agente=DDQN(max_memory=memory_len,lr=learning_rate,epsilon_decay=epsilon_decay,epsilon_min=epsilon_min,model_load=load)
current_state=env.reset_env()
current_state=np.reshape(current_state,[1,current_state.shape[0]])

# Llenando  la memoria
for i in range(memory_len):
    action=agente.choose_action(current_state)
    accion=lista_acciones[action]
    next_state,reward,done,lavel=env.step(action=accion)
    next_state=np.reshape(next_state,[1,next_state.shape[0]])
    agente.remember(Transition(current_state,reward,action,next_state,done))
    if (lavel != 'Playing'):
        current_state=env.reset_env()
        current_state=np.reshape(current_state,[1,current_state.shape[0]])
    else:
        current_state=next_state
        
print('Memory Full')
for i in range(1,episodio):
    current_state=env.reset_env()
    current_state=np.reshape(current_state,[1,current_state.shape[0]])
    score = 0
    while True:
        action=agente.choose_action(current_state)
        accion=lista_acciones[action]
        next_state,reward,done,lavel=env.step(action=accion)
        next_state=np.reshape(next_state,[1,next_state.shape[0]])
        agente.remember(Transition(current_state,reward,action,next_state,done))
        show_map(env)
        score += reward
        if (lavel != 'Playing'):
            if i % 10 == 0:
                print('Total reward {} game statement {} Episodie {}/{} '.format(round(score,2),lavel,i,episodio))
                agente.save_model()
            break
        current_state=next_state
        agente.replay(batch_memory=batch_size)
pg.quit()

