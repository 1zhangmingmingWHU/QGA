"""
This code contains implementations that reference or adapt work from other sources.
Acknowledgements:
- https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py
- https://github.com/nicklashansen/tdmpc2

The original sources have been modified and adapted for the specific needs of this project.
"""

import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from copy import deepcopy
import numpy as np
import os
from typing import List


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 756)
        self.l2 = nn.Linear(756, 756)
        self.l3 = nn.Linear(756, action_dim)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, state):
        a = F.relu(self.l1(state.to(self.device)))
        a = F.relu(self.l2(a))
        return self.l3(a)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Set the number of bins and value range for quantizing Q-values
        self.num_bins = 101
        self.vmin = 0
        self.vmax = +5
        self.bin_size = (self.vmax - self.vmin) / self.num_bins
        # Define two separate critic networks
        self.l1 = nn.Linear(state_dim + action_dim, 756)
        self.l2 = nn.Linear(756, 756)
        self.l3 = nn.Linear(756, self.num_bins)

        self.l4 = nn.Linear(state_dim + action_dim, 756)
        self.l5 = nn.Linear(756, 756)
        self.l6 = nn.Linear(756, self.num_bins)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def symexp(self,x):
        """
        Symmetric exponential function.
        """
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

    def two_hot_inv(self,x):
        """
        Converts a batch of soft two-hot encoded vectors to scalars.

        This function takes a two-hot encoded representation and converts it back to a scalar Q-value.
        The two-hot encoding is used to represent scalar values by distributing the probability mass
        between two adjacent bins, which helps avoid quantization errors when mapping continuous values
        to discrete categories. By applying a weighted sum over the bin values, this function recovers
        the original scalar Q-value for use in subsequent calculations, ensuring a more accurate
        representation of the Q-value during training.
        """
        if self.num_bins == 0:
            return x
        elif self.num_bins == 1:
            return self.symexp(x)
        DREG_BINS = torch.linspace(self.vmin, self.vmax, self.num_bins, device=x.device)
        x = F.softmax(x, dim=-1)
        x = torch.sum(x * DREG_BINS, dim=-1, keepdim=True)
        return self.symexp(x)

    def forward(self, state, action,return_type: int=0):
        sa = torch.cat([state.to(self.device), action.to(self.device)], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        if return_type == 1:
            return q1, q2

        Q1, Q2 = self.two_hot_inv(q1), self.two_hot_inv(q2)
        return Q1, Q2


    def Q1(self, state, action):
        sa = torch.cat([state.to(self.device), action.to(self.device)], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3_BC(nn.Module):
    def __init__(self, dim_obs, action_dim=1,discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5,
                 policy_freq=2, alpha=2.5):
        super(TD3_BC, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(dim_obs, action_dim).to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(dim_obs, action_dim).to(self.device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha


        self.total_it = 0
        self.num_bins = 101  # 原值为101
        self.vmin = 0
        self.vmax = +5
        self.bin_size = (self.vmax - self.vmin) / self.num_bins

    def take_actions(self, states: torch.Tensor) -> np.ndarray:
        states = states.type(torch.float).to(self.device)
        actions = self.actor(states)  # Pass device as argument
        actions = actions.cpu().data.numpy()
        return actions

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)
        action = self.actor(state)
        return action

    def symlog(self,x):
        """
        Symmetric logarithmic function.
        """
        return torch.sign(x) * torch.log(1 + torch.abs(x))

    def two_hot(self,x):
        """
        Converts a batch of scalars to soft two-hot encoded targets for discrete regression.

        The two-hot encoding is used to distribute the probability mass between two adjacent bins,
        which ensures that scalar targets are represented more accurately. This method avoids the
        errors that come from discretizing continuous targets into a single bin. By placing probability
        on two neighboring bins, it captures the underlying value more precisely,
        leading to better gradient signals during training.

        This method comes from the work of Google DeepMind, as described in the paper
        《Stop Regressing: Training Value Functions via Classification for Scalable Deep RL》.
        """
        if self.num_bins == 0:
            return x
        elif self.num_bins == 1:
            return self.symlog(x)
        x = torch.clamp(self.symlog(x), self.vmin, self.vmax).squeeze(1)
        bin_idx = torch.floor((x - self.vmin) / self.bin_size).long()
        bin_offset = ((x - self.vmin) / self.bin_size - bin_idx.float()).unsqueeze(-1)
        soft_two_hot = torch.zeros(x.size(0), self.num_bins, device=x.device)
        soft_two_hot.scatter_(1, bin_idx.unsqueeze(1), 1 - bin_offset)
        soft_two_hot.scatter_(1, (bin_idx.unsqueeze(1) + 1) % self.num_bins, bin_offset)
        return soft_two_hot

    def soft_ce(self,pred, target):
        """
        Computes the cross entropy loss between predictions and soft targets.

        The target is converted to a soft two-hot encoded representation, which helps the network
        learn a smoother representation of the value function. This indicates that using categorical
        cross-entropy instead of mean squared error for value targets can significantly enhance stability and performance in RL.
        """
        pred = F.log_softmax(pred, dim=-1)
        target = self.two_hot(target)
        return -(target * pred).sum(-1, keepdim=True)

    def step(self,state, action, reward, next_state, done):
        self.total_it += 1

        state, action, reward, next_state, not_done = state, action, reward, next_state, 1 - done

        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            )

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action,return_type=1)
        critic_loss=self.soft_ce(current_Q1, target_Q).mean()+self.soft_ce(current_Q2,target_Q).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        # Update actor network at a reduced frequency
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            Q = self.critic.Q1(state, pi)
            lmbda = self.alpha / Q.abs().mean().detach()

            actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.update_target(self.critic, self.critic_target)
            self.update_target(self.actor, self.actor_target)

        return critic_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy() if actor_loss is not None else None

    def update_target(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_((1. - self.tau) * target_param.data + self.tau * local_param.data)

    def save_jit(self, save_path: str):
        '''
        save model as JIT
        '''
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        scripted_policy = torch.jit.script(self.cpu())
        scripted_policy.save(save_path + "/td3_bc_model" + ".pth")


if __name__ == '__main__':
    # Example usage
    state_dim = 3
    action_dim = 1
    max_action = 1.0

    model = TD3_BC(state_dim, action_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print("Learnable parameters: {:,}".format(total_params))

    # Simulate environment interactions and fill replay buffer
    step_num = 1000
    batch_size = 1000
    for i in range(1000):
        state = np.random.uniform(2, 5, size=(state_dim,))
        next_state = np.random.uniform(2, 5, size=(state_dim,))
        action = np.random.uniform(-1, 1, size=(action_dim,))
        reward = np.random.uniform(0, 1, size=(1,))
        done = np.random.choice([0, 1], size=(1,))
        state, next_state, action, reward, terminal = torch.tensor(state, dtype=torch.float), torch.tensor(
            next_state, dtype=torch.float), torch.tensor(action, dtype=torch.float), torch.tensor(reward,
                                                                                                    dtype=torch.float), torch.tensor(
            done, dtype=torch.float)

        q_loss, a_loss = model.step(state, action, reward, next_state, done)
        print(f'step:{i} q_loss:{q_loss} a_loss:{a_loss}')

