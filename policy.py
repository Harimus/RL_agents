
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import torch.nn.functional as F
import gym

"""This file contains the Neural network policies. The reason for this being in an independent file is so that
    All the different Policy Gradient methods can be used with the same policy network, for performance comparison.
    Why? Because one might want to understand the effect of policy network/depth/size on the network."""


def fcnn_policy(layer_sizes, activation_function=nn.Tanh, output_activation=nn.Identity):
    """
    This is a basic fully connected neural network, used as an default for the basic model
    @param layer_sizes: a list of integers that specify each hidden layers size
    @param activation_function: Specify the activation function for the hidden layers.
    @param output_activation: The activation function for the output layer.
    @return: nn.Sequential
    """
    nn_layers = []
    for j in range(len(layer_sizes)-1):
        activation = activation_function if j < len(layer_sizes)-2 else output_activation
        nn_layers += [nn.Linear(layer_sizes[j], layer_sizes[j+1]), activation()]
    return nn.Sequential(*nn_layers) # Star "unpacks" the list, in to args for the function


class BasicPolicy(nn.Module):

    def __init__(self, observation_space, action_space, hidden_layers=(30, 20),  network=fcnn_policy,
                 activation=nn.Tanh, output_activation=nn.Identity, discrete=True):
        super().__init__()
        self.policy = network([observation_space]+list(hidden_layers)+[action_space],
                              activation_function=activation, output_activation=output_activation)
        self.discrete_action = discrete
        log_std = -0.5 * np.ones(action_space, dtype=np.float32)
        self.log_std = torch.nn.Parameter( torch.as_tensor(log_std))
    def get_policy(self, obs):
        logits = self.policy(obs)
        if self.discrete_action:
            return Categorical(logits=logits)
        else:
            return Normal(logits, self.log_std.exp())


    def forward(self, obs: torch.Tensor, act=None):
        pi = self.get_policy(obs)

        log_act = None
        if act is not None:
            log_act = pi.log_prob(act)
        return pi, log_act

    def step(self, obs):
        with torch.no_grad():
            pi = self.get_policy(obs)
            a = pi.sample()
            logp_a = pi.log_prob(a)

        return a.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class ActorCriticPolicy(nn.Module):

    def __init__(self, observation_space, action_space, hidden_layer=(30, 20),
                 actor=BasicPolicy, critic=fcnn_policy, discrete=True):
        super().__init__()
        # The policy, "actor"
        self.pi = actor(observation_space, action_space, hidden_layers=hidden_layer, discrete=discrete)
        # The critic
        self.v = critic([observation_space]+list(hidden_layer)+[1])

    def step(self, obs):
        with torch.no_grad():
            a, log_p = self.pi.step(obs)
            v = self.v(obs)
            return a, log_p, v.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class FCNNQFunction(nn.Module):
    def __init__(self, observation_space, action_space, hidden_layers, activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()
        self.network = fcnn_policy([observation_space.shape[0] + action_space.n]+list(hidden_layers)+[1], activation_function=activation, output_activation=output_activation)

    def forward(self, observation, action):
        q = self.network(torch.cat([observation, action], dim=-1))
        return torch.squeeze(q, -1)


LOG_STD_MIN = 2
LOG_STD_MAX = 20


class SquashedGaussianFCNNPolicy(nn.Module):
    def __init__(self, observation_space, action_space, hidden_layers, act_max, act_min, activation=nn.ReLU):
        super().__init__()
        self.net = fcnn_policy([observation_space] + list(hidden_layers),
                               activation_function=activation, output_activation=activation)
        self.mu = nn.Linear(hidden_layers[-1], action_space)
        self.log_std= nn.Linear(hidden_layers[-1], action_space)
        self.act_scale = (act_max+act_min)
        self.act_const = act_min

    def forawrd(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu(net_out)
        log_std = self.log_std(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        pi_dist = Normal(mu, std)
        if deterministic:
            pi_act = mu
        else:
            pi_act = pi_dist.rsample()
        if with_logprob:
            logp_pi = pi_dist.log_prob(pi_act).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_act - F.softplus(-2 * pi_act))).sum(axis=1)
        else:
            logp_pi = None
        pi_act = torch.tanh(pi_act)
        pi_act = self.act_scale * pi_act + self.act_const
        return pi_act, logp_pi

"""The actor critic policy resembles the """
class SACActorCriticPolicy(nn.Module):
    def __init__(self, observation_space, action_space, hidden_layer=(30, 20), actor=SquashedGaussianFCNNPolicy,
                 critic=FCNNQFunction):
        super().__init__()
        self.pi = actor(observation_space.shape[0], action_space.shape[0], hidden_layers=hidden_layer,
                        activation=nn.ReLU)
        self.q1 = critic(observation_space, action_space, hidden_layer, activation=nn.ReLU)
        self.q2 = critic(observation_space, action_space, hidden_layer, activation=nn.ReLU)


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    obs_space = env.observation_space.shape[0]
    act_space = env.action_space.n
    state = env.reset()
    state_as_torch = torch.as_tensor(state, dtype=torch.float32)
    acp = ActorCriticPolicy(obs_space, act_space)
    bp = BasicPolicy(obs_space, act_space)

    action, _, value = acp.step(state_as_torch)
    action2, _ = bp.step(state_as_torch)
    print("ActorCritic: ", action)
    print("Actor: ", action2)
    print("Critic (value func for state): ", value)
    env.step(action)
    env.step(action2)