import numpy as np
import scipy.optimize as opt
import torch
import time
import torch.nn as nn
from torch.nn.parameter import Parameter
from collections.abc import MutableMapping


# Post processors ==================================================
def identity(rews, obs, acts):
    return rews

def madodiv(rews, obs, acts):
    return rews/variation_dim(obs, order=1)

def variodiv(rews, obs, acts):
    return rews/variation_dim(obs, order=2)

def radodiv(rews, obs, acts):
    return rews/variation_dim(obs, order=.5)


def mdim_div(rews, obs, acts):
    if obs.shape[0] == 1000:
        gait_start = 200
        m, _, _, _ = mesh_dim(obs[gait_start:])
        m = np.clip(m, 1, obs.shape[1] / 2)
    else:
        m = obs.shape[1] / 2

    return rews / m


def cdim_div(rews, obs, acts):
    if obs.shape[0] == 1000:
        gait_start = 200
        _, c, _, _ = mesh_dim(obs[gait_start:])
        c = np.clip(c, 1, obs.shape[1] / 2)
    else:
        c = obs.shape[1] / 2

    return rews / c


# Rollout Fns ==================================================
def do_rollout(env, policy, render=False):
    torch.autograd.set_grad_enabled(False)

    act_list = []
    obs_list = []
    rew_list = []

    dtype = torch.float32
    obs = env.reset()
    done = False
    cur_step = 0

    while not done:
        obs = torch.as_tensor(obs, dtype=dtype).detach()
        obs_list.append(obs.clone())

        act = policy(obs)
        obs, rew, done, _ = env.step(act.numpy())
        if render:
            env.render()
            time.sleep(.01)

        act_list.append(torch.as_tensor(act.clone()))
        rew_list.append(rew)

        cur_step += 1

    ep_length = len(rew_list)
    ep_obs = torch.stack(obs_list)
    ep_act = torch.stack(act_list)
    ep_rew = torch.tensor(rew_list, dtype=dtype)
    ep_rew = ep_rew.reshape(-1, 1)

    torch.autograd.set_grad_enabled(True)
    return ep_obs, ep_act, ep_rew, ep_length


def do_long_rollout(env, policy, ep_length):
    torch.autograd.set_grad_enabled(False)

    act_list = []
    obs_list = []
    rew_list = []

    dtype = torch.float32
    obs = env.reset()
    cur_step = 0

    while cur_step < ep_length:
        obs = torch.as_tensor(obs, dtype=dtype).detach()
        obs_list.append(obs.clone())

        act = policy(obs)
        obs, rew, done, _ = env.step(act.numpy())

        act_list.append(torch.as_tensor(act.clone()))
        rew_list.append(rew)

        cur_step += 1

    ep_length = len(rew_list)
    ep_obs = torch.stack(obs_list)
    ep_act = torch.stack(act_list)
    ep_rew = torch.tensor(rew_list, dtype=dtype)
    ep_rew = ep_rew.reshape(-1, 1)

    torch.autograd.set_grad_enabled(True)
    return ep_obs, ep_act, ep_rew, ep_length


# Policy =======================================================================
class MLP(nn.Module):
    """
    Policy to be used with [redacted]s rl module.
    Simple MLP that has a linear layer at the output
    """

    def __init__(
            self, input_size, output_size, num_layers, layer_size, activation=nn.ReLU, output_activation=nn.Identity,
            bias=True, input_bias=None, dtype=torch.float32):
        """
         input_size: how many inputs
         output_size: how many outputs
         num_layers: how many HIDDEN layers
         layer_size: how big each hidden layer should be
         activation: which activation function to use
         input_bias: can add an "input bias" such that the first layer computer activation([x+bi]*W^T + b0)
         """
        super(MLP, self).__init__()

        self.activation = activation()
        self.output_activation = output_activation()

        if input_bias is not None:
            self.input_bias = Parameter(torch.Tensor(input_size))
        else:
            self.input_bias = None

        if num_layers == 0:
            self.layers = []
            self.output_layer = nn.Linear(input_size, output_size, bias=bias)

        else:
            self.layers = nn.ModuleList([nn.Linear(input_size, layer_size, bias=bias)])
            self.layers.extend([nn.Linear(layer_size, layer_size, bias=bias) for _ in range(num_layers - 1)])
            self.output_layer = nn.Linear(layer_size, output_size, bias=bias)

        self.state_means = torch.zeros(input_size, requires_grad=False)
        self.state_std = torch.ones(input_size, requires_grad=False)

    def forward(self, data):

        if self.input_bias is not None:
            data += self.input_bias

        data = (torch.as_tensor(data) - self.state_means) / self.state_std

        for layer in self.layers:
            data = self.activation(layer(data))

        return self.output_activation(self.output_layer(data))

    def to(self, place):
        super(MLP, self).to(place)
        self.state_means = self.state_means.to(place)
        self.state_std = self.state_std.to(place)
        return self



class BoxMesh(MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, d):
        self.d = d; self.scale = 1/d
        self.mesh = dict()

    def __getitem__(self, key):
        return self.mesh[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.mesh[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.mesh[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.mesh)

    def __len__(self):
        return len(self.mesh)

    def __keytransform__(self, key):
        round_key = np.asarray(key)
        round_key = np.round(round_key * self.scale, decimals=0) * self.d
        round_key[round_key == -0.0] = 0.0
        round_key = tuple(round_key)
        return round_key

    

# Dimensionality calculations ==================================================
def create_box_mesh(data, d, initial_mesh=None):
    """ Creates a mesh from the given data using boxes of size d
    Args:
        data: np.array, the data you want to create a mesh for
        d: float, the length of the box used to determine membership in the mesh
        initial_mesh: dict, output from a previous call to create_box_mesh, used to add to a mesh rather than build a new one
    Returns:
        mesh: dict, keys are the mesh point coordinates, values are how many points in the original data set are represented by the mesh point
    """
    if initial_mesh is None:
        initial_mesh = {}

    mesh = initial_mesh
    data = np.asarray(data)

    scale = 1 / d

    keys = np.round(data * scale, decimals=0) * d
    keys[keys == -0.0] = 0.0

    for key in keys:
        key = tuple(key)
        if key in mesh:
            mesh[key] += 1
        else:
            mesh[key] = 1

    return mesh


def mesh_dim(data, scaling_factor=1.5, init_d=1e-2, upper_size_ratio=4/5, lower_size_ratio=0.0, d_limit=1e-9):
    """
    Args:
        data - any array like, represents the trajectory you want to compute the dimension of
        scaling factors - float indicating how much to scale d by for every new mesh
        init_d - float, initial box size
        upper_size_ratio - upper_size_ratio*data.shape[0] determines what size of mesh to stop at when finding the upper bound of the curve.
        lower_size_ratio - lower_size_ratio*data.shape[0] determines what size of mesh to stop at when finding the lower bound of the curve. Usually best to leave at 0.
        d_limit - smallest d value to allow when seeking the upper_size bound

    Returns:
        mdim: linear fit to the log(mesh) log(d) data, intentional underestimate of the meshing dimensions
        cdim: the conservative mesh dimension, that is the largest slope from the log log data, an intentional overestimate of the
        mesh_sizes: sizes of each mesh created during the computation
        d_vals: box sizes used to create each mesh during the computation
    """

    mesh_size_upper = np.round(upper_size_ratio * data.shape[0])
    mesh_size_lower = np.round(np.max((1.0, lower_size_ratio * data.shape[0])))
    d = init_d

    mesh = create_box_mesh(data, d)
    mesh_sizes = [len(mesh)]
    d_vals = [d]

    while mesh_sizes[0] < mesh_size_upper and d > d_limit:
        d /= scaling_factor
        mesh = create_box_mesh(data, d)
        mesh_sizes.insert(0, len(mesh))
        d_vals.insert(0, d)

    d = init_d
    while mesh_sizes[-1] > mesh_size_lower and d > d_limit:
        d = d * scaling_factor
        mesh = create_box_mesh(data, d)
        mesh_sizes.append(len(mesh))
        d_vals.append(d)

    for i, m in enumerate(mesh_sizes):
        if m < mesh_size_upper:
            lin_begin = i
            break

    xdata = np.log2(d_vals[lin_begin:])
    ydata = np.log2(mesh_sizes[lin_begin:])

    # Fit a curve to the log log line
    def f(x, m, b):
        return m * x + b

    popt, pcov = opt.curve_fit(f, xdata, ydata)

    # find the largest slope
    min_slope = 0
    for i in range(len(ydata) - 2):
        slope = (ydata[i+1] - ydata[i]) / (xdata[i + 1] - xdata[i])
        if slope < min_slope:
            min_slope = slope

    return -popt[0], -min_slope, mesh_sizes, d_vals


def variation_dim(X, order=1):
    # Implements the order p variation fractal dimension from https://arxiv.org/pdf/1101.1444.pdf (eq 18)
    # order 1 corresponds to the madogram, 2 to the variogram, 1/2 to the rodogram
    return 2 - 1 / (order * np.log(2)) * (np.log(power_var(X, 2, order)) - np.log(power_var(X, 1, order)))


def power_var(X, l, ord):
    # The power variation, see variation_dim
    diffs = X[l:] - X[:-l]
    norms = np.zeros(diffs.shape[0])
    for i, d in enumerate(diffs):
        norms[i] = np.linalg.norm(d, ord=1)

    return 1 / (2 * len(X) - l) * np.sum(norms)

