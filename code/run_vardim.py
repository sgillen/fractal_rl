from ars import ars, postprocess_default
from common import *
import copy
import gym
import time
import xarray as xr
import numpy as np
import os

env_names = ["HalfCheetah-v2", "Hopper-v2", "Walker2d-v2"]
post_fns = [identity, variodiv, madodiv]

torch.set_default_dtype(torch.float64)
num_experiments = len(post_fns)
num_seeds = 10
num_epochs = 750
n_workers = 24
n_delta = 60
n_top = 20
exp_noise =.025

save_dir = "./data_test0/"
env_config = {}

os.makedirs(f"{save_dir}")

start = time.time()
for env_name in env_names:
    env = gym.make(env_name, **env_config)
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.shape[0]
    policy_dict = {fn.__name__: [] for fn in post_fns}

    rewards = xr.DataArray(np.zeros((num_experiments, num_seeds, num_epochs)),
                           dims=("post", "trial", "epoch"),
                           coords={"post": [fn.__name__ for fn in post_fns]})

    post_rewards = xr.DataArray(np.zeros((num_experiments, num_seeds, num_epochs)),
                                dims=("post", "trial", "epoch"),
                                coords={"post": [fn.__name__ for fn in post_fns]})

    data = xr.Dataset(
        {"rews": rewards,
         "post_rews": post_rewards},
        coords={"post": [fn.__name__ for fn in post_fns]},
        attrs={"policy_dict": policy_dict, "post_fns": post_fns, "env_name": env_name,
               "hyperparams": {"num_experiments": num_experiments, "num_seeds": num_seeds, "num_epochs": num_epochs,
                               "n_workers": n_workers, "n_delta": n_delta, "n_top": n_top, "exp_noise": exp_noise},
               "env_config": env_config})

    for post_fn in post_fns:
        for i in range(num_seeds):
            policy = MLP(in_size, out_size, 0, 0)
            policy, r_hist, lr_hist = ars(env_name, policy, num_epochs, n_workers=n_workers, n_delta=n_delta,
                                          n_top=n_top, exp_noise=exp_noise, postprocess=post_fn,
                                          env_config=env_config)
            print(f"{env_name}, {post_fn.__name__}, {i}, {time.time() - start}")
            data.policy_dict[post_fn.__name__].append(copy.deepcopy(policy))
            data.rews.loc[post_fn.__name__, i, :] = lr_hist
            data.post_rews.loc[post_fn.__name__, i, :] = r_hist

    torch.save(data, f"{save_dir}/{env_name}.xr")
