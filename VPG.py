import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import gym
from RLbasics.targets import monte_carlo_target
import matplotlib.pyplot as plt
from policy import ActorCriticPolicy
import scipy.signal
import numpy as np


def moving_avg(x, w):
    return np.convolve(x.squeeze(), np.ones(w), 'valid')/w


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def gae(episodes, values, gamma, lam):
    """Generalized Advantage Estimator"""
    rewards = [ep[2] for ep in episodes]
    deltas =np.array(rewards) + gamma * np.array(values[1:]+[0]) - np.array(values)
    advantage =  discount_cumsum(deltas, gamma*lam)
    return advantage


def vpg(policy=ActorCriticPolicy, epoch=10000, gamma=0.99, lam=0.97,
        actor_lr=1e-3, critic_lr=1e-3):
    env = gym.make('Pendulum-v0') #'MountainCar-v0')
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    agent = policy(observation_space, action_space, discrete=False)
    actor_optimizer = Adam(agent.pi.parameters(), lr=actor_lr)
    critic_optimizer = Adam(agent.v.parameters(), lr=critic_lr)

    def compute_actor_loss(episodes, adv, logp_old):
        states = [ep[0] for ep in episodes]
        actions = [ep[1] for ep in episodes]
        pi, log_pi = agent.pi(torch.as_tensor(states, dtype=torch.float32), torch.as_tensor(actions, dtype=torch.float32))
        loss_pi = -(log_pi*torch.as_tensor(list(adv), dtype=torch.float32)).mean()

        approx_kl = (torch.as_tensor(np.array(logp_old), dtype=torch.float32) - log_pi).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)
        return loss_pi, pi_info

    def compute_critic_loss(episodes):
        states = [ep[0] for ep in episodes]
        rewards = [ep[2] for ep in episodes]
        rewards_to_go = discount_cumsum(rewards, gamma)
        return ((agent.v(torch.as_tensor(states, dtype=torch.float32)) - torch.as_tensor(list(rewards_to_go), dtype=torch.float32))**2).mean()



    fig, ax = plt.subplots()
    ax.set_title(env.spec._env_name + " total reward per episode")
    ax.set_ylabel("reward")
    ax.set_xlabel("episodes")
    ax.set_xlim([0, epoch])
    ax.set_ylim([-2000, 0])
    rewards_per_episode=[]
    percent = 0
    for i in range(1, epoch+1):
        done = False
        trajectory = []
        values  = []
        state = env.reset()
        cum_reward = 0
        log_p = []
        log_p_old = []
        while not done:
            state_as_torch = torch.as_tensor(state, dtype=torch.float32)
            action, lp, val = agent.step(state_as_torch)

            if(i == epoch-1):
                env.render()
            next_state, reward, done, _ = env.step([action])
            episode = (state, action.item(), reward, next_state)
            trajectory.append(episode)
            values.append(val[0])
            log_p.append(lp)
            cum_reward += reward
        if not log_p_old:
            log_p_old = log_p
        advantage = gae(trajectory, values, gamma, lam)
        actor_optimizer.zero_grad()
        loss_pi, pi_info = compute_actor_loss(trajectory, advantage, log_p_old)
        loss_pi.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        loss_v = compute_critic_loss(trajectory)
        loss_v.backward()
        critic_optimizer.step()
        rewards_per_episode.append(cum_reward)
        log_p_old = log_p
        if i % (epoch / 100) == 0:
            percent += 1
            print(percent, "% done.", ' loss_pi: ', loss_pi.item(), ' loss_v: ', loss_v.item(),  ' approx kl: ', pi_info['kl'], ' entropy: ', pi_info['ent'])
            if (percent % 10 ) == 0:
                w = 5
                plot_rewards= moving_avg(np.array(rewards_per_episode), w)
                ax.plot(np.array(range(1, len(plot_rewards)+1)), plot_rewards)
                fig.show()


if __name__ == '__main__':
    ag = vpg()
