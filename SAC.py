import torch
from torch.optim import Adam
import gym
import random
from copy import deepcopy
from collections import deque
import torch.nn as nn
import numpy as np
from torch.distributions import Distribution, Independent, Normal
from policy import fcnn_policy
import matplotlib.pyplot as plt
from util import plot_mountain_cart
from tqdm import tqdm
"""Model defined here too, and follows Kaixhin structure rather than Spinningup,
since it is easier to understand"""

# TODO: Add reward scaling.
# TODO:
import os
steps, rewards = [], []
fig, ax = plt.subplots()


def plot(step, reward, title, epoch=1000000):
    steps.append(step)
    rewards.append(reward)
    ax.plot(steps, rewards, 'b-')
    ax.set_title(title)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Rewards')
    ax.set_xlim((0, epoch))
    ax.set_ylim((0, 10000))
    fig.savefig('./'+title + '.png')

class TanhNormal(Distribution):
    """Copied from Kaixhi"""
    def __init__(self, loc, scale):
        super().__init__()
        self.normal = Independent(Normal(loc, scale), 1)

    def sample(self):
        return torch.tanh(self.normal.sample())

    # samples with re-parametrization trick (differentiable)
    def rsample(self):
        return torch.tanh(self.normal.rsample())

    # Calculates log probability of value using the change-of-variables technique
    # (uses log1p = log(1 + x) for extra numerical stability)
    def log_prob(self, value):
        inv_value = (torch.log1p(value) - torch.log1p(-value)) / 2  # artanh(y)
        # log p(f^-1(y)) + log |det(J(f^-1(y)))|
        return self.normal.log_prob(inv_value) - torch.log1p(-value.pow(2) + 1e-6).sum(dim=1)

    @property
    def mean(self):
        return torch.tanh(self.normal.mean)

    def get_std(self):
        return self.normal.stddev


LOG_MIN = -20
LOG_MAX = 2


class SoftActor(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_layer_size=(256, 256),
                 activation_function=nn.Tanh, output_activation=nn.Identity):
        super().__init__()
        """The network have output size of action_dim*2 because output of this network are 
        divided in to the mu and log_std component of a gaussian distribution"""
        self.network = fcnn_policy([observation_dim]+list(hidden_layer_size)+[2*action_dim],
                                   activation_function=activation_function, output_activation=output_activation)

    def forward(self, state: torch.Tensor):
        mu, log_std = self.network(state).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_MIN, LOG_MAX)  # to make it not too random/deterministic
        normal = TanhNormal(mu, log_std.exp())
        return normal


class Critic(nn.Module):
    def __init__(self, observation_dim, action_dim=None, hidden_layer_size=(256, 256),
                 activation_function=nn.Tanh, output_activation=nn.Identity):
        super().__init__()
        network_architecture = [observation_dim + (action_dim if action_dim is not None else 0)] + list(hidden_layer_size) + [1]
        self.network = fcnn_policy(network_architecture,
                                   activation_function=activation_function, output_activation= output_activation)

    def forward(self, state, action=None):
        if action is None:
            value = self.network(state)
        else:
            value = self.network(torch.cat([state, action], dim=-1))
        return value.squeeze(dim=-1)


class SoftActorCritic(nn.Module):

    def __init__(self, observation_dim, action_dim, hidden_layer_size=(256, 256),
                 activation_function=nn.ReLU, output_activation=nn.Identity):
        super().__init__()
        self.actor = SoftActor(observation_dim=observation_dim, action_dim=action_dim,
                               hidden_layer_size=hidden_layer_size, activation_function=activation_function,
                               output_activation=output_activation)
        self.critic_v = Critic(observation_dim=observation_dim, action_dim=None,
                               hidden_layer_size=hidden_layer_size, activation_function=activation_function,
                               output_activation=output_activation)

        self.critic_q1 = Critic(observation_dim=observation_dim, action_dim=action_dim,
                                hidden_layer_size=hidden_layer_size, activation_function=activation_function,
                                output_activation=output_activation)

        self.critic_q2 = Critic(observation_dim=observation_dim, action_dim=action_dim,
                                hidden_layer_size=hidden_layer_size, activation_function=activation_function,
                                output_activation=output_activation)
        self.target_critic_v = deepcopy(self.critic_v)
        for param in self.target_critic_v.parameters():
            param.requires_grad = False
        self.name = "Soft Actor Critic"

    def forward(self, state):
        return self.actor(state)


def sac(policy=SoftActorCritic, epoch=100000, gamma=0.99, polyak=0.995,
        actor_lr=1e-3, critic_lr=1e-3, alpha=0.2, buffer_size=100000,
        off_policy_batch_size=128, reward_scale=1, initial_exploration=10000,
        update_interval=1, test_interval=1000, in_agent=None):

    envname = 'Humanoid-v2'#'MountainCarContinuous-v0' #'Pendulum-v0
    env = gym.make(envname)


    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    # Scaling of action
    scaling_factor = (env.action_space.high[0] - env.action_space.low[0])/2
    scaling_const = (env.action_space.high[0] + env.action_space.low[0])/2

    def scale_action(act):
        scaled_act= scaling_factor*act + scaling_const
        if (scaled_act > env.action_space.high[0])[0] or (scaled_act < env.action_space.low[0])[0]:
            print("wtf: ", act)
        return np.clip(scaling_factor*act + scaling_const,
                        env.action_space.low[0], env.action_space.high[0])

    def test(actor, render=False):
        with torch.no_grad():
            actor.eval()
            state, total_reward, done = env.reset(), 0, False

            while not done:
                action = actor(torch.as_tensor(state, dtype=torch.float32)).mean
                env.render() if render is True else _
                state, reward, done, _ = env.step(scale_action(action.numpy()))
                total_reward += reward
            #env.close() #crashes mujoco_py
            #plot_mountain_cart(agent) if render else _
            actor.train()
        return total_reward
    if in_agent is not None:
        agent = in_agent
    else:
        agent = policy(observation_space, action_space, hidden_layer_size=(32, 32), activation_function=nn.Tanh)

    actor_optimizer = Adam(agent.actor.parameters(), lr=actor_lr)
    critic_q_optimizer = Adam(list(agent.critic_q1.parameters())+list(agent.critic_q2.parameters()), lr=critic_lr)
    critic_v_optimizer = Adam(agent.critic_v.parameters(), lr=critic_lr)
    replay_buffer = deque(maxlen=buffer_size)

    def update(episodes, update_target=False):
        states = torch.as_tensor([ep[0] for ep in episodes], dtype=torch.float32)
        actions = torch.stack([ep[1] for ep in episodes])
        #actions = actions.unsqueeze(-1)
        rewards = torch.as_tensor([ep[2] for ep in episodes], dtype=torch.float32)
        next_states = torch.as_tensor([ep[3] for ep in episodes], dtype=torch.float32)
        is_done = torch.as_tensor([ep[-1] for ep in episodes], dtype=torch.float32)

        # Target for V
        pi = agent.actor(states)
        sample_actions = pi.rsample()
        entropy_component = alpha * pi.log_prob(sample_actions)
        critic_q = torch.min(agent.critic_q2(states, sample_actions),
                             agent.critic_q1(states, sample_actions)).detach()
        target_v = critic_q - entropy_component.detach()

        # Update V

        critic_v_loss = (agent.critic_v(states) - target_v).pow(2).mean()
        critic_v_optimizer.zero_grad()
        critic_v_loss.backward()
        critic_v_optimizer.step()

        # Target for Q
        target_q = rewards + gamma * (1-is_done) * agent.target_critic_v(next_states)
        # Update Q
        critic_q_loss = (agent.critic_q1(states, actions) - target_q).pow(2).mean() + \
                        (agent.critic_q2(states, actions) - target_q).pow(2).mean()
        critic_q_optimizer.zero_grad()
        critic_q_loss.backward()
        critic_q_optimizer.step()


        # Update Policy
        target_pi = torch.min(agent.critic_q2(states, sample_actions), agent.critic_q1(states, sample_actions))
        loss_pi = (entropy_component - target_pi).mean()
        actor_optimizer.zero_grad()
        loss_pi.backward()
        actor_optimizer.step()
        # Update target V
        for V_param, target_V_param in zip(agent.critic_v.network.parameters(), agent.target_critic_v.network.parameters()):
            target_V_param.data = polyak * target_V_param.data + (1 - polyak) * V_param.data

    state, done = env.reset(), False
    progress_bar = tqdm(range(1, epoch+1), unit_scale=1, smoothing=0)
    for i in progress_bar:
        with torch.no_grad():
            if i < initial_exploration:
                # uniform distribution action taken for better exploration at the beginning roughtly 10%
                action = torch.tensor(np.random.rand(action_space)*2 - 1, dtype=torch.float32)
            else:
                action = agent(torch.as_tensor(state, dtype=torch.float32)).sample()
            next_state, reward, done, _ = env.step(scale_action(action.numpy()))
            replay_buffer.append([state, action, reward_scale*reward, next_state, done])
            if done:
                state = env.reset()
            else:
                state = next_state
        if i > initial_exploration and i % update_interval == 0:
            batch = random.sample(replay_buffer, off_policy_batch_size)
            update(batch, True)

        if i > initial_exploration and i % test_interval == 0:
            total_reward = test(agent, render=True)# if i % (epoch/10) else False)
            plot(i, total_reward, "SoftActorCritic", epoch)
            if i % (epoch / 100) == 0:
                progress_bar.set_description('Step: %i | Reward: %f' % (i, total_reward))
    plt.show()
    return agent

if __name__ == '__main__':
    #ag = sac(epoch=10000, reward_scale=1, alpha=0.4)
    ag = sac(alpha=0.8, off_policy_batch_size=400)

    #ag = sac(in_agent=ag)
