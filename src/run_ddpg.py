import gym
import numpy as np
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TD3
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import pybullet_envs
import time
import os
from stable_baselines.common import make_vec_env
from multiprocessing import Process
import seagul.envs.bullet
import json
from shutil import copyfile
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env.vec_normalize import VecNormalize

import gym
import numpy as np
from numpy import pi
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines import TD3
from multiprocessing import Process
from stable_baselines.common import make_vec_env
import os
import time

import pickle
import torch
import signal

num_steps = int(2e5)
base_dir = os.path.dirname(os.path.abspath(__file__)) + "/data_td3_c2/"
trial_name = input("Trial name: ")
exp_dir = base_dir + trial_name + "/"
base_ok = input("run will be saved in " + exp_dir + " ok? y/n")

if base_ok == "n":
    exit()

def run_stable(num_steps, save_dir, env_name):    
    env = make_vec_env(env_name, n_envs=1, monitor_dir=save_dir)
    env = VecNormalize(env)
    n_actions = env.action_space.shape[-1]
    #n_actions = 6
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    
    model = TD3(MlpPolicy,
                env,
                action_noise=action_noise,
                verbose=1,
                gamma = 0.99,
                buffer_size= 1000000,
                learning_starts= 10000,
                batch_size= 100,
                learning_rate= 1e-3,
                train_freq= 1000,
                gradient_steps= 1000,
                policy_kwargs={"layers":[400, 300]},
                n_cpu_tf_sess=1,
    )

    model.learn(total_timesteps=num_steps)
    model.save(save_dir + "/model.zip")

if __name__ == "__main__":
    for env_name in ['HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']:
              
        trial_dir = exp_dir + env_name
        start = time.time()
        proc_list = []
        
        os.makedirs(trial_dir, exist_ok=False)
            
        for seed in np.random.randint(0, 2 ** 32, 10):
                
            save_dir = trial_dir + "/" + str(seed)
            os.makedirs(save_dir, exist_ok=False)
            
            #run_stable(num_steps, save_dir)
            p = Process(
                target=run_stable,
                args=(num_steps, save_dir, env_name)
            )
            p.start()
            proc_list.append(p)
            
        for p in proc_list:
            print("joining")
            p.join()

    print(f"experiment complete, total time: {time.time() - start}, saved in {save_dir}")
