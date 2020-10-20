import gym
import numpy as np
from numpy import pi
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from multiprocessing import Process
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
import os
import time
import ipdb

import pickle
import torch
import signal

num_steps = int(2e6)
base_dir = os.path.dirname(os.path.abspath(__file__)) + "/data_ppo_c2/"
trial_name = input("Trial name: ")
exp_dir = base_dir + trial_name + "/"
base_ok = input("run will be saved in " + exp_dir + " ok? y/n")

if base_ok == "n":
    exit()


def run_stable(num_steps, save_dir, env_name):
    env = make_vec_env(env_name, n_envs=10, monitor_dir=save_dir)
    env = VecNormalize(env)
    
    model = PPO2(MlpPolicy,
                 env,
                 verbose=0,
                 seed=int(seed),
                 # normalize = True
                 # policy = 'MlpPolicy',
                 n_steps=1024,
                 nminibatches=64,
                 lam=0.95,
                 gamma=0.99,
                 noptepochs=10,
                 ent_coef=0.0,
                 learning_rate=2.5e-4,
                 cliprange=0.1,
                 cliprange_vf=-1,
                 )

    num_epochs = 5

    for epoch in range(num_epochs):

        model.learn(total_timesteps=int(num_steps/num_epochs))
        model.save(save_dir + "/model.zip")
        env.save(save_dir + "/vec_env.zip")


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

