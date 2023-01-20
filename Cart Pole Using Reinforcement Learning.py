#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import gym
from stable_baselines3 import PPO
#stablebaselines is easy to use and implement,it will make the reinforcement learning easy accessible to audience.
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
#evaluate policy helps in finding how the model will be performing


# In[3]:


environment_name='CartPole-v1'


# In[4]:


env=gym.make(environment_name)


# In[5]:


episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()


# In[6]:


env.action_space#action return 2 values i.e 0 and 1,0 for pushing cart to the left and 1 for pushing cart to the right.


# In[7]:


env.reset()
env.step(1)


# In[8]:


env.action_space.sample()


# In[9]:


env.observation_space#it return 4 values i.e cart position,cart velocity,pole angle,pole angular velocity


# In[10]:


env.observation_space.sample()


# In[11]:


log_path=os.path.join('Training','Logs')


# In[12]:


log_path


# In[13]:


env=gym.make(environment_name)
env=DummyVecEnv([lambda:env])
model=PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)
#stable baseline3 has three policy:MlpPolicy,CnnPolicy,MultiInputPolicy


# In[14]:


#train the model
model.learn(total_timesteps=20000)


# In[15]:


PPO_path=os.path.join("Training","Saved Models","PPO_Model_Cartpole")


# In[16]:


model.save(PPO_path)


# In[17]:


PPO_path


# In[18]:


model.learn(total_timesteps=1000)


# In[19]:


evaluate_policy(model,env,n_eval_episodes=10,render=True)


# In[20]:


env.close()


# In[21]:


#test our model
episodes = 5
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0 
    while not done:
        env.render()
        action,_=model.predict(obs)
        obs, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))


# In[22]:


action


# In[23]:


env.close()

