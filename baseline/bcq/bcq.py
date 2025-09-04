"""
This code contains implementations that reference or adapt work from other sources.
Acknowledgements:
- https://github.com/sfujim/BCQ/blob/master/continuous_BCQ/BCQ.py
- https://github.com/nicklashansen/tdmpc2

The original sources have been modified and adapted for the specific needs of this project.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, phi=0.05):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 1400)
        self.l2 = nn.Linear(1400, 1300)
        self.l3 = nn.Linear(1300, action_dim)

        self.max_action = max_action
        self.phi = phi

    def forward(self, state, action):
        a = F.mish(self.l1(torch.cat([state, action], 1)))
        a = F.mish(self.l2(a))
        a = self.phi * self.max_action * torch.tanh(self.l3(a))
        # a = self.phi * self.max_action * self.l3(a)
        return (a + action).clamp(-self.max_action, self.max_action)

# Define the Critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,num_bins,vmin,vmax):
        super(Critic, self).__init__()
        # Number of bins for quantizing Q-values
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.bin_size = (self.vmax - self.vmin) / self.num_bins

        # Critic network 1
        self.l1 = nn.Linear(state_dim + action_dim, 1400)
        self.l2 = nn.Linear(1400, 1300)
        self.l3 = nn.Linear(1300, self.num_bins)

        # Critic network 2 (for double Q-learning)
        self.l4 = nn.Linear(state_dim + action_dim, 1400)
        self.l5 = nn.Linear(1400, 1300)
        self.l6 = nn.Linear(1300, self.num_bins)

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
        # Create bins for discrete regression
        DREG_BINS = torch.linspace(self.vmin, self.vmax, self.num_bins, device=x.device)
        # Apply softmax to ensure valid probability distribution
        x = F.softmax(x, dim=-1)
        # Weighted sum over bins to get scalar value
        x = torch.sum(x * DREG_BINS, dim=-1, keepdim=True)
        return self.symexp(x)

    def forward(self, state, action,return_type: int=0):
        q1 = F.mish(self.l1(torch.cat([state, action], 1)))
        q1 = F.mish(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.mish(self.l4(torch.cat([state, action], 1)))
        q2 = F.mish(self.l5(q2))
        q2 = self.l6(q2)

        # Return raw logits if requested
        if return_type == 1:
            return q1, q2

        # Convert logits to scalar Q-values
        Q1, Q2 = self.two_hot_inv(q1), self.two_hot_inv(q2)
        return Q1, Q2

    def q1(self, state, action,return_type: int=0):
        q1 = F.mish(self.l1(torch.cat([state, action], 1)))
        q1 = F.mish(self.l2(q1))
        q1 = self.l3(q1)

        # Return raw logits if requested
        if return_type == 1:
            return q1

        # Convert logits to scalar Q-values
        Q1= self.two_hot_inv(q1)
        return Q1

# Define the Vanilla Variational Auto-Encoder
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super(VAE, self).__init__()
        # Encoder layers
        self.e1 = nn.Linear(state_dim + action_dim, 1750)
        self.e2 = nn.Linear(1750, 1750)

        # Latent space parameters
        self.mean = nn.Linear(1750, latent_dim)
        self.log_std = nn.Linear(1750, latent_dim)

        # Decoder layers
        self.d1 = nn.Linear(state_dim + latent_dim, 1750)
        self.d2 = nn.Linear(1750, 1750)
        self.d3 = nn.Linear(1750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        # Encode state and action into latent space
        z = F.mish(self.e1(torch.cat([state, action], 1)))
        z = F.mish(self.e2(z))

        # Compute mean and log standard deviation for latent distribution
        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        # Reparameterize to sample from latent space
        z = mean + std * torch.randn_like(std)
        # Decode latent vector back to action
        u = self.decode_withz(state, z, apply_tanh=False)
        return u, mean, std

    def decode_withz(self, state, z, apply_tanh=True):
        a = F.mish(self.d1(torch.cat([state, z], 1))).clamp(-0.5, 0.5)
        a = F.mish(self.d2(a))
        a = self.d3(a)
        if apply_tanh:
            return self.max_action * torch.tanh(a)
        else:
            return a  # 线性输出

    def decode(self, state, apply_tanh=True):
        z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)
        a = F.mish(self.d1(torch.cat([state, z], 1)))
        a = F.mish(self.d2(a))
        a = self.d3(a)
        if apply_tanh:
            return self.max_action * torch.tanh(a)
        else:
            return a

# Define the BCQ model
class BCQ(nn.Module):
    def __init__(self, state_dim=3, action_dim=1, max_action=200,discount=0.99, tau=0.005, lmbda=0.75, phi=0.05):
        latent_dim = action_dim * 2
        super().__init__()
        # Number of bins for quantizing Q-values
        self.num_bins = 101  # 原值为101
        self.vmin = 0
        self.vmax = +5
        self.bin_size = (self.vmax - self.vmin) / self.num_bins

        self.actor = Actor(state_dim, action_dim, max_action, phi).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim,self.num_bins,self.vmin,self.vmax).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, device).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters())

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.lmbda = lmbda
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        # target，也就是reward只有正数
        pred = F.log_softmax(pred, dim=-1)
        target = self.two_hot(target)
        return -(target * pred).sum(-1, keepdim=True)

    def forward(self, states: torch.Tensor):
        # Get the best action for a given state using the Actor network
        with torch.no_grad():
            state = states.reshape(1, -1).repeat(100, 1).to(self.device)
            action = self.actor(state, self.vae.decode(state))
            q1 = self.critic.q1(state, action)
            ind = q1.argmax(0)
        return action[ind].flatten()

    def take_actions(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(self.device)
            action = self.actor(state, self.vae.decode(state))
            q1 = self.critic.q1(state, action)
            ind = q1.argmax(0)
        return action[ind].cpu().data.numpy().flatten()

    def step(self, state, action, reward, next_state, terminal):
        # Variational Auto-Encoder Training
        batch_size=state.shape[0]
        recon, mean, std = self.vae(state, action)
        recon_loss = F.mse_loss(recon, action)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss

        # Backpropagate VAE loss
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        # Critic Training
        with torch.no_grad():
            # Duplicate next state 10 times
            next_state = torch.repeat_interleave(next_state, 10, 0)

            # Compute value of perturbed actions sampled from the VAE
            target_Q1, target_Q2 = self.critic_target(next_state,
                                                      self.actor_target(next_state, self.vae.decode(next_state, apply_tanh=True)))

            # Soft Clipped Double Q-learning
            target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1,
                                                                                                    target_Q2)
            # Take max over each action sampled from the VAE
            target_Q = target_Q.reshape(batch_size, -1).max(1)[0].reshape(-1, 1)
            target_Q = reward + (1-terminal) * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action,return_type=1)
        critic_loss = self.soft_ce(current_Q1, target_Q).mean()+self.soft_ce(current_Q2,target_Q).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Pertubation Model / Action Training
        sampled_actions = self.vae.decode(state)
        perturbed_actions = self.actor(state, sampled_actions)

        # Update through DPG
        actor_loss = -self.critic.q1(state, perturbed_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Target Networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss,actor_loss, vae_loss

    def save_jit(self, save_path: str):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # Script each submodule individually
        #torch.save(self.cpu(),save_path + "/bcq_model" + ".pth")
        scripted_policy = torch.jit.script(self.cpu())
        scripted_policy.save(save_path + "/bcq_model" + ".pth")


if __name__ == '__main__':
    pass
