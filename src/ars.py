import gym
import torch
from multiprocessing import Process,Pipe
from common import MLP
import os
import numpy as np

def postprocess_default(rews, obs, acts):
    return rews


def ars(env_name, policy, n_epochs, env_config={}, n_workers=8, step_size=.02, n_delta=32, n_top=16, exp_noise=0.03, zero_policy=True, postprocess=postprocess_default):
    #torch.multiprocessing.set_sharing_strategy('file_system')    
    torch.autograd.set_grad_enabled(False)
    """
    Augmented Random Search
    https://arxiv.org/pdf/1803.07055
    
    modified to apply a post processing function to the reward of each episode

    Args:
        env_name: string with the registered OpenAI gym environment to train on
        policy: the policy to train, assumed to be an MLP from above
        n_epochs: how many epochs to train for
        env_config: kwargs for the environment
        n_workers: how many workers to use during training
        step_size: alpha from the original paper
        n_delta: N in original paper, how many deltas to explore in each epoch
        n_top: b in original paper, how many deltas to keep each epoch to do updates with
        exp_noise: noise used for exploration
        zero_policy: if true will zero out policy parameters before training, as in the original paper, else will train the given policy as is
        postprocess: function to apply to rewards after every episode 

    Returns:
        policy: the trained policy
        r_hist: history of rewards after postprocessing 
        lr_hist: history of rewards before postprocessing was applied

    Example:
        torch.set_default_dtype(torch.float64)
        from seagul.nn import MLP
    
        env_name = "HalfCheetah-v2"
        env = gym.make(env_name)
        in_size = env.observation_space.shape[0]
        out_size = env.action_space.shape[0]
    
        policy = MLP(in_size, out_size, 0, 0, bias=False)
    
        policy, r_hist, lr_hist = ars(env_name, policy, 20, n_workers=8, n_delta=32, n_top=16)

    
    """

#    import ipdb; print(f"main_thread:{os.getpid()})"); ipdb.set_trace()
    
    proc_list = []
    master_pipe_list = []

    for i in range(n_workers):
        master_con, worker_con = Pipe()
        proc = Process(target=worker_fn, args=(worker_con, env_name, env_config, policy, postprocess))
        proc.start()
        proc_list.append(proc)
        master_pipe_list.append(master_con)

 #   import ipdb; print(f"main_thread:{os.getpid()})"); ipdb.set_trace()

    W = torch.nn.utils.parameters_to_vector(policy.parameters())
    n_param = W.shape[0]

    if zero_policy:
        W = torch.zeros_like(W)

    env = gym.make(env_name,**env_config)
    s_mean = policy.state_means
    s_std = policy.state_std
    total_steps = 0
    env.close()

    r_hist = []
    lr_hist = []

    exp_dist = torch.distributions.Normal(torch.zeros(n_delta, n_param), torch.ones(n_delta, n_param))

    for epoch in range(n_epochs):

        deltas = exp_dist.sample()
        pm_W = torch.cat((W+(deltas*exp_noise), W-(deltas*exp_noise)))

        for i, Ws in enumerate(pm_W):
            master_pipe_list[i % n_workers].send((Ws,s_mean,s_std))

        results = []
        for i, _ in enumerate(pm_W):
            results.append(master_pipe_list[i % n_workers].recv())
        
        states = torch.empty(0)
        p_returns = []
        m_returns = []
        l_returns = []
        top_returns = []

        for p_result, m_result in zip(results[:n_delta], results[n_delta:]):
            ps, pr, plr = p_result
            ms, mr, mlr = m_result

            states = torch.cat((states, ms, ps), dim=0)
            p_returns.append(pr)
            m_returns.append(mr)
            l_returns.append(plr); l_returns.append(mlr)
            top_returns.append(max(pr,mr))

        top_idx = sorted(range(len(top_returns)), key=lambda k: top_returns[k], reverse=True)[:n_top]
        p_returns = torch.stack(p_returns)[top_idx]
        m_returns = torch.stack(m_returns)[top_idx]
        l_returns = torch.stack(l_returns)[top_idx]

        lr_hist.append(l_returns.mean())
        r_hist.append((p_returns.mean() + m_returns.mean())/2)

        ep_steps = states.shape[0]
        s_mean = update_mean(states, s_mean, total_steps)
        s_std = update_std(states, s_std, total_steps)
        total_steps += ep_steps
        
        if epoch % 5 == 0:
            print(f"epoch: {epoch}, reward: {lr_hist[-1].item()}, processed reward: {r_hist[-1].item()} ")

        W = W + (step_size / (n_delta * torch.cat((p_returns, m_returns)).std() + 1e-6)) * torch.sum((p_returns - m_returns)*deltas[top_idx].T, dim=1)


    #import ipdb; print(f"main_thread:{os.getpid()})"); ipdb.set_trace()
    for pipe in master_pipe_list:
        pipe.send("STOP")

    for proc in proc_list:
        proc.join()
        
    policy.state_means = s_mean
    policy.state_std = s_std
    torch.nn.utils.vector_to_parameters(W, policy.parameters())
    return policy, r_hist, lr_hist


def update_mean(data, cur_mean, cur_steps):
    new_steps = data.shape[0]
    return (torch.mean(data, 0) * new_steps + cur_mean * cur_steps) / (cur_steps + new_steps)


def update_std(data, cur_std, cur_steps):
    new_steps = data.shape[0]
    batch_var = torch.var(data, 0)

    if torch.isnan(batch_var).any():
        return cur_std
    else:
        cur_var = cur_std ** 2
        new_var = torch.std(data, 0) ** 2
        new_var[new_var < 1e-6] = cur_var[new_var < 1e-6]
        return torch.sqrt((new_var * new_steps + cur_var * cur_steps) / (cur_steps + new_steps))


def worker_fn(worker_con, env_name, env_config, policy, postprocess):

    env = gym.make(env_name, **env_config)
    epoch = 0

    while True:
        data = worker_con.recv()

        if data == "STOP":
            print(f"worker {os.getpid()} closing shop")
            env.close()
            return
        else:
            #print(f"worker {os.getpid()} got rollout request")
            W,state_mean,state_std = data

            policy.state_std = state_std
            policy.state_means = state_mean

            states, returns, log_returns = do_rollout_train(env, policy, postprocess, W)
            #print(f"worker {os.getpid()} rollout finished sending return")
            worker_con.send((np.array(states), np.array(returns), np.array(log_returns)))
            #print(f"worker {os.getpid()} return sent")
            epoch +=1


def do_rollout_train(env, policy, postprocess, W, obs_std=0, act_std=0):
    torch.nn.utils.vector_to_parameters(W, policy.parameters())    
    state_list = []
    act_list = []
    reward_list = []

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    obs = torch.as_tensor(env.reset())
    done = False
    while not done:
        state_list.append(obs)

        actions = policy(obs)
        actions += torch.randn(act_size)*act_std

        obs, reward, done, _ = env.step(np.clip(np.asarray(actions), -1, 1))
        obs = torch.as_tensor(obs)
        obs += torch.randn(obs_size)*obs_std


        act_list.append(torch.as_tensor(actions))
        reward_list.append(reward)

    state_tens = torch.stack(state_list)
    act_tens = torch.stack(act_list)
    preprocess_sum = torch.as_tensor(sum(reward_list))
    nstate_tens = (state_tens - policy.state_means) / policy.state_std
    reward_list = postprocess(torch.tensor(reward_list), nstate_tens, act_tens)
    reward_sum = torch.as_tensor(sum(reward_list))

    return state_tens, reward_sum, preprocess_sum



if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    from seagul.nn import MLP

    env_name = "HalfCheetah-v2"
    env = gym.make(env_name)
    in_size = env.observation_space.shape[0]
    out_size = env.action_space.shape[0]

    policy = MLP(in_size, out_size, 0, 0, bias=False)

    policy, r_hist, lr_hist = ars(env_name, policy, 20, n_workers=8, n_delta=32, n_top=16)
