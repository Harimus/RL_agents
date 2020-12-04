import torch
import gym
import random
from collections import deque
import torch.nn as nn
import numpy as np
from torch.distributions import Distribution, Independent, Normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""Made for mountain Cart continuous"""
def plot_Q_value(ax, agent_critic_Q, states, actions):
    X, Y, Z, Q = [], [], [], []
    with torch.no_grad():
        for x in states[0]:
            for y in states[1]:
                for z in actions:
                    X.append(x), Y.append(y), Z.append(z)
        in_state = torch.as_tensor(list(zip(X, Y)), dtype=torch.float32)
        in_action = torch.as_tensor(Z, dtype=torch.float32).unsqueeze(-1)
        Q = agent_critic_Q(in_state, in_action).numpy()

    return ax.scatter(X, Y, Z, c=Q, cmap=plt.hot())

def plot_V_value(ax, agent_critic_V, states):
    X, Y = np.meshgrid(states[0], states[1])
    XY = np.dstack((X, Y))
    XY_torch = torch.as_tensor(XY, dtype=torch.float32)
    with torch.no_grad():
        Z_torch = agent_critic_V(XY_torch)
        Z = Z_torch.squeeze(-1).numpy()
    return ax.plot_surface(X, Y, Z)

def plot_policy(ax, agent_policy, states):
    X, Y = np.meshgrid(states[0], states[1])
    XY = np.dstack((X, Y))
    XY_torch = torch.as_tensor(XY, dtype=torch.float32)
    with torch.no_grad():
        pi = agent_policy(XY_torch)
        mu_torch = pi.mean
        mu = mu_torch.squeeze(-1).numpy()
        std = pi.get_std().squeeze(-1).numpy()
    return ax.scatter(X.reshape(-1), Y.reshape(-1), mu.reshape(-1), c=std.reshape(-1), cmap=plt.hot())

def plot_mountain_cart(agent):
    fig = plt.figure('Montuain Cart')
    fig.suptitle('Mountain Cart (Continous) for ' + agent.name)
    env = gym.make('MountainCarContinuous-v0')
    min_pos, min_vel = env.observation_space.low
    max_pos, max_vel = env.observation_space.high
    min_acc, max_acc = env.action_space.low[0], env.action_space.high[0]
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.set_title('Value(V) function')
    ax1.set_xlabel('Position[m]')
    ax1.set_xlim((min_pos, max_pos))
    ax1.set_ylabel('Velocity [m/s]')
    ax1.set_ylim((min_vel, max_vel))
    ax1.set_zlabel('V(s) value')
    num_points = 100
    states = np.linspace(min_pos, max_pos, num_points), np.linspace(min_vel, max_vel, num_points)
    plot_V_value(ax1, agent.critic_v, states)

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_title('Action-value(Q) function')
    ax2.set_xlabel('Position[m]')
    ax2.set_xlim((min_pos, max_pos))
    ax2.set_ylabel('Velocity [m/s]')
    ax2.set_ylim((min_vel, max_vel))
    ax2.set_zlabel('Action value')
    actions = np.linspace(min_acc, max_acc, 10)
    cbar_Q = fig.colorbar(plot_Q_value(ax2, agent.critic_q1, states, actions))
    cbar_Q.ax.set_ylabel('Q value')
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.set_title('Policy (color=std)')
    ax3.set_xlabel('Position[m]')
    ax3.set_xlim((min_pos, max_pos))
    ax3.set_ylabel('Velocity [m/s]')
    ax3.set_ylim((min_vel, max_vel))
    ax3.set_zlabel('action')
    ax3.set_zlim(min_acc, max_acc)
    cbar_pi = fig.colorbar(plot_policy(ax3, agent.actor, states))
    cbar_pi.ax.set_ylabel('std')


def save_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion, epochs, replay_buffer, filename='./saved_model.pth'):
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
        'replay_buffer': replay_buffer
                },  filename)


def load_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, filename='./saved_model.pth'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['loss'], checkpoint['epoch'], checkpoint['replay_buffer']


if __name__ == '__main__':
    from SAC import SoftActorCritic
    envname = 'MountainCarContinuous-v0'
    env = gym.make(envname)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    agent = SoftActorCritic(observation_space, action_space)
    plot_mountain_cart(agent)
    plt.show()