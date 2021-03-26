# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 16:27:23 2021

@author: MBI
"""
import tensorflow as tf 
import tensorflow_probability as tfp 
import threading,gym,time
import matplotlib.pyplot as plt
import concurrent.futures as cf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(1)

#%%

class A3C():
    def __init__(self,env,lr_actor,lr_critic):
        self.env = env
        self.memory = []
        self.beta = 0.9
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.state_dim = env.observation_space.shape[0]
        self.state = self.env.reset()
        self.state = tf.expand_dims(tf.cast(self.state,dtype=tf.float32),0)
        self.lock = threading.Lock()
        self.loss_actor = []
        self.loss_critic = []
        self.build_autoencoder()
    
    def reset_memory(self):
        self.memory=[]
        
    def remember(self,item):
        self.memory.append(item)
    
    def build_autoencoder(self):
        # Encoder
        inputs = tf.keras.Input(shape=(self.state_dim,))
        feature_size = 32
        
        X = tf.keras.layers.Dense(units=256,activation='relu')(inputs)
        X = tf.keras.layers.Dense(units=128,activation='relu')(X)
        output_enco = tf.keras.layers.Dense(units=feature_size)(X)
        self.encoder = tf.keras.Model(inputs,output_enco,name="Encoder")

        # Decoder
        feature_inputs = tf.keras.Input(shape=(feature_size,))
        Y = tf.keras.layers.Dense(units=128,activation='relu')(feature_inputs)
        Y = tf.keras.layers.Dense(units=256,activation='relu')(Y)
        output_deco = tf.keras.layers.Dense(units=self.state_dim)(Y)
        self.decoder = tf.keras.Model(feature_inputs,output_deco)

        # Autoencoder
        self.autoencoder = tf.keras.Model(inputs,self.decoder(self.encoder(inputs)))
        self.autoencoder.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=3e-3),loss='mse')
    def train_autoencoder(self,x_train,y_test):
        self.autoencoder.fit(x_train,x_train,validation_data=(y_test,y_test),batch_size=32,epochs=10)
        
    
    def log(self,arg):
        mean,std,action = arg
        std = tf.keras.activations.softplus(x=std) + 1e-10
        dist = tfp.distributions.Normal(loc=mean[0],scale=std[0])
        lgp = dist.log_prob(action)
        return lgp
    
    def entropy(self,arg):
        mean,std = arg
        std = tf.keras.activations.softplus(x=std) + 1e-10
        dist = tfp.distributions.Normal(loc=mean[0],scale=std[0])
        entropy = dist.entropy()
        return entropy
    
    def action(self,arg):
        mean,std = arg
        std = tf.keras.activations.softplus(x=std) + 1e-10
        dist = tfp.distributions.Normal(loc=mean[0],scale=std[0])
        action_dist = dist.sample(1)
        action = tf.clip_by_value(action_dist,self.env.action_space.low[0],self.env.action_space.high[0])
        return action
    
    def logp_loss(self,entropy,beta=0.0):
        def loss(y_true,y_pred):
           return -tf.keras.backend.mean((y_true*y_pred)+(beta*entropy),axis=1)
        return loss
    
    def get_entropy(self,state):
        entropy = self.entropy_model.predict(state)[0]
        return entropy
    
    
    def build_model(self):
        Inputs = tf.keras.Input(shape=(self.state_dim,))
        self.encoder.trainable = False
        x = self.encoder(Inputs)
        mean = tf.keras.layers.Dense(units=1,kernel_initializer='zero',name='Mean')(x)
        std = tf.keras.layers.Dense(units=1,kernel_initializer='zero',name='Std')(x)
        action = tf.keras.layers.Lambda(self.action,output_shape=(1,))([mean,std])
        log_prob = tf.keras.layers.Lambda(self.log,output_shape=(1,))([mean,std,action])
        entropy = tf.keras.layers.Lambda(self.entropy,output_shape=(1,))([mean,std])
        valor = tf.keras.layers.Dense(units=1,kernel_initializer='zero',name='Valor')(x)
        
        # Modelos
        self.entropy_model = tf.keras.Model(inputs=Inputs,outputs=entropy)
        self.log_model = tf.keras.Model(inputs=Inputs,outputs=log_prob)
        self.Actor = tf.keras.Model(inputs=Inputs,outputs=action)
        self.Critic = tf.keras.Model(inputs=Inputs,outputs=valor)
        
        # Perdidas 
        loss = self.logp_loss(self.get_entropy(self.state),self.beta)
        optimizer_valor = tf.keras.optimizers.Adam(learning_rate=self.lr_critic)
        optimizer_actor = tf.keras.optimizers.RMSprop(learning_rate=self.lr_actor)
        self.log_model.compile(optimizer=optimizer_actor,loss=loss)
        self.Critic.compile(optimizer=optimizer_valor,loss='mse')

        # Sumario
        try:
            tf.keras.utils.plot_model(self.Actor,to_file='C:/Users/MBI/Documents/Python_Scripts/Practicas_AI/MountainCarContinouse/Actor.png', show_shapes=True,show_layer_names=True,dpi=100)
            tf.keras.utils.plot_model(self.Critic,to_file='C:/Users/MBI/Documents/Python_Scripts/Practicas_AI/MountainCarContinouse/Critic.png', show_shapes=True,show_layer_names=True,dpi=100)
            print("[+] Estructura guardadas.")
        except :
            print("[-] Error en estructuras.")
    
    def save_weights(self):
        try:
            self.Actor.save_weights('C:/Users/MBI/Documents/Python_Scripts/Practicas_AI/MountainCarContinouse/Actor_weights.h5')
            self.Critic.save_weights('C:/Users/MBI/Documents/Python_Scripts/Practicas_AI/MountainCarContinouse/Critic_weights.h5')
        except :
            raise ValueError('[-] Error saving weights')
    
    def load_weights(self):
        try:
            self.Actor.load_weights('C:/Users/MBI/Documents/Python_Scripts/Practicas_AI/MountainCarContinouse/Actor_weights.h5')
            self.Critic.load_weights('C:/Users/MBI/Documents/Python_Scripts/Practicas_AI/MountainCarContinouse/Critic_weights.h5')
            print('[+] Loaded weights')
        except :
            print('[-] Error loading weights')
    
    def save_model(self):
        try:
            self.Actor.save("C:/Users/MBI/Documents/Python_Scripts/Practicas_AI/MountainCarContinouse/A3C_model.h5")
            print("[+] Model saved.")
        except :
            print("[-] Error model not saved.")
    
    
    def load_encode_weights(self):
        self.encoder.load_weights("C:/Users/MBI/Documents/Python_Scripts/Practicas_AI/MountainCarContinouse/Encode_weights.h5")
    
    def saved_encoder(self):
        try:
            self.encoder.save_weights("C:/Users/MBI/Documents/Python_Scripts/Practicas_AI/MountainCarContinouse/Encode_weights.h5")
            print("[+] Encoder saved")
        except:
            raise("[-] Error to save encoder")
        
    
    def choose_action(self,observation):
        state = tf.expand_dims(tf.cast(observation,dtype=tf.float32),0)
        action = self.Actor.predict(state)[0]
        return action
    
     
    def train_by_episode(self,last_value,gamma=1.0):
        r = last_value
        for item in self.memory[::-1]:
            [step,state,next_state,reward,done] = item
            r = reward + gamma * r
            state = tf.expand_dims(tf.cast(state,dtype=tf.float32),0)
            item = [step,state,next_state,r,done]
            self.train(item,gamma)

        self.save_weights()
    
    def train(self,item,gamma):
        [step,state,next_state,reward,done] = item
        self.state = state
        discount_factory = gamma**step
        delta = reward - self.Critic.predict(state)[0]
        discount_delta = discount_factory*delta 
        discount_delta = tf.expand_dims(tf.cast(discount_delta,dtype=tf.float32),0)
        history_actor = self.log_model.train_on_batch(state,discount_delta)
        discount_delta = reward
        discount_delta =  tf.expand_dims(tf.cast(discount_delta,dtype=tf.float32),0)
        history_critic = self.Critic.train_on_batch(state,discount_delta)
        self.loss_actor.append(history_actor)
        self.loss_critic.append(history_critic)
    

#%%
def show_historial(loss_actor,loss_critic,rewards,episodes):
    plt.figure(figsize=(16,8))
    fig,(ax_1,ax_2) = plt.subplots(1,2)
    ax_1.plot(loss_actor,color='r')
    ax_1.plot(loss_critic,color='b')
    ax_1.set_xlabel('Episodes')
    ax_1.set_ylabel('Losses')
    ax_1.legend(['Losses_Actor','Losses_Critic'])
    ax_2.plot(rewards)
    ax_2.set_xlabel('Episodes')
    ax_2.set_ylabel('Rewards')
    plt.savefig('C:/Users/MBI/Documents/Python_Scripts/Practicas_AI/MountainCarContinouse/A3C.png',dpi=100)
    
    plt.show()
#%%

episode = 500 - 400
env=gym.make('MountainCarContinuous-v0')
env.reward_range = (-1,1)
learnig_rate_actor = 3e-3
learnig_rate_critic = 1e-3
gamma = 0.99
agent = A3C(env,lr_actor=learnig_rate_actor,lr_critic=learnig_rate_critic)
try:
    agent.load_encode_weights()
    print("[+] Encode  weights loaded.")
except :
    print('[*] Weights not found / sample env...')
    x_train = [env.observation_space.sample() for i in range(200000)]
    y_test = [env.observation_space.sample() for i in range(200000)]
    x_train = tf.cast(x_train,dtype=tf.float32)
    y_test = tf.cast(y_test,dtype=tf.float32)
    agent.train_autoencoder(x_train,y_test)
    agent.saved_encoder()

agent.build_model()
try:
    agent.load_weights()
except : pass

rewards = []
def train_agent(env):
    global score
    score = 0.0
    for i in range(1,episode):
        state = env.reset()
        done = False
        step = 0
        agent.reset_memory()
        while not done:
            action=agent.choose_action(state)
            next_state,reward,done,_= env.step(action)
            item = [step, state, next_state, reward, done]
            agent.remember(item)
            score += reward
            state = next_state
            step += 1
            env.render() 
            if done:
                v = 0 if reward > 0 else agent.Critic(tf.expand_dims(tf.cast(next_state,dtype=tf.float32),0))[0]
                agent.lock.acquire() 
                agent.train_by_episode(v,gamma)
                agent.lock.release() 
                print('[*] Episode: {}/{} Total Rewards: {}'.format(i,episode,round(score,0))) 
        rewards.append(score)
    env.close()

def train_workers(n_workers):
    envs=[gym.make('MountainCarContinuous-v0') for  i in range(n_workers)]
    with cf.ThreadPoolExecutor(max_workers=n_workers) as workers:
        hilo = []
        for i in range(n_workers):
            hilo.append(workers.submit(train_agent,envs[i]))
            time.sleep(1)

    for h in cf.as_completed(hilo):
        h.done()
#%%
train_workers(8)
#%%
show_historial(agent.loss_actor,agent.loss_critic,rewards,episode)
#%%
agent.save_model()
#%%
# Seccion de Prueba del modelo.Nota la fucion action tiene que existir donde de carge el modelo por el problema de serializacion de las Capas lambdas.
# De esta manera logro delimitar la seccion de entrenamiento de la de prueba.

env = gym.make('MountainCarContinuous-v0')

def action(arg):
    mean,std = arg
    std = tf.keras.activations.softplus(x=std) + 1e-10
    dist=tfp.distributions.Normal(loc=mean[0],scale=std[0])
    action_dist=dist.sample(1)
    action=tf.clip_by_value(action_dist,env.action_space.low[0],env.action_space.high[0])
    return action

def load_model():
    try:
       model = tf.keras.models.load_model("C:/Users/MBI/Documents/Python_Scripts/Practicas_AI/MountainCarContinouse/A3C_model.h5",custom_objects={"action":action})
       print("[+] Model loaded.")
       return model
    except:
         raise("[-] Erro load model.")

def test_model():
    scores = []
    model = load_model()
    for i in range(100):
        score = 0
        done = False
        state = env.reset()
        state = tf.expand_dims(tf.convert_to_tensor(state,dtype=tf.float32),axis=0)
        while  not done:
            action = model.predict(state)[0]
            next_state,reward,done,_ = env.step(action)
            score += reward
            env.render()
            
            if done:
                print('[*] Episode: {}/{} Total Rewarsd: {}'.format(i,100,round(score,0)))

            state = tf.expand_dims(tf.convert_to_tensor(next_state,dtype=tf.float32),axis=0)
        scores.append(score)
    env.close()
    
    promedio = round(sum(scores)/100)
    
    if promedio >= 110:
        print('[*] Victoty score: {}'.format(score))
    else:
        print('[-] Defeat score: {}'.format(score))

#%%
test_model()
#%%
env.close()
#%%







