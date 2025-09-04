"""
This code contains implementations that reference or adapt work from other sources.
Acknowledgements:
- https://github.com/aviralkumar2907/CQL

The original sources have been modified and adapted for the specific needs of this project.
"""

import os
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Policy Network Definition
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor, reparameterize: bool = True):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc3(x)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)

        # Sample actions using reparameterization trick
        if reparameterize:
            eps = torch.randn_like(std)
            actions = mean + eps * std
        else:
            actions = mean

        return actions

    def compute_log_prob(self, actions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc3(x)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)

        # Calculate log probability under Gaussian distribution
        log_prob = -0.5 * (
            ((actions - mean) / (std + 1e-6)) ** 2 + 2 * log_std + torch.log(torch.tensor(2 * torch.pi))
        ).sum(-1, keepdim=True)
        return log_prob


class CQL(nn.Module):
    '''
    CQL model with Expectile Regression and Enhanced Action Sampling
    '''

    def __init__(self, dim_obs=2, dim_actions=1, gamma=1, tau=0.001, V_lr=1e-4, critic_lr=1e-4, actor_lr=1e-4,
                 network_random_seed=1, expectile=0.7, temperature=1.0):
        super().__init__()
        self.num_of_states = dim_obs #state维度
        self.num_of_actions = dim_actions #action维度
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(network_random_seed)
        np.random.seed(network_random_seed)

        # Initialize networks
        # 用的是SAC算法，所以这里有五个网络
        self.policy = PolicyNetwork(dim_obs, dim_actions) # 策略网络
        self.qf1 = self.build_network(dim_obs + dim_actions, 1) # 价值网络1 输入维度：state和action的维度相加 输出维度：1
        self.qf2 = self.build_network(dim_obs + dim_actions, 1) # 价值网络2
        self.target_qf1 = self.build_network(dim_obs + dim_actions, 1) # 目标价值网络1
        self.target_qf2 = self.build_network(dim_obs + dim_actions, 1) # 目标价值网络2

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=actor_lr)
        self.qf1_optimizer = optim.Adam(self.qf1.parameters(), lr=critic_lr)
        self.qf2_optimizer = optim.Adam(self.qf2.parameters(), lr=critic_lr)

        self.target_entropy = -dim_actions # 目标熵的大小
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device) # 使用alpha的log值,可以使训练结果比较稳定
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=actor_lr)
        self.min_q_weight = 1.0

        self.to(self.device)

    def build_network(self, input_dim: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, output_dim),
        )

    def step(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, dones: torch.Tensor):
        '''
        train model
        '''
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Policy and Alpha Loss
        new_obs_actions = self.policy(states, reparameterize=True) # 计算下一个action
        log_pi=self.policy.compute_log_prob(new_obs_actions,states) # 下一个动作的对数概率

        # Update alpha (temperature parameter)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean() #熵正则项alpha的损失
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        alpha = self.log_alpha.exp()

        q_new_actions = torch.min( # 用两个critic计算新动作的Q值
            self.qf1(torch.cat([states, new_obs_actions], dim=-1)),
            self.qf2(torch.cat([states, new_obs_actions], dim=-1))
        )

        policy_loss = (alpha * log_pi - q_new_actions).mean() #actor网络的损失
        self.policy_optimizer.zero_grad() #更新actor即策略网络
        policy_loss.backward()
        self.policy_optimizer.step()

        # QF Loss
        q1_pred = self.qf1(torch.cat([states, actions], dim=-1))
        q2_pred = self.qf2(torch.cat([states, actions], dim=-1))

        with torch.no_grad():
            next_actions = self.policy(next_states, reparameterize=True)
            next_log_pi = self.policy.compute_log_prob(next_actions, next_states)
            target_q_values = torch.min(
                self.target_qf1(torch.cat([next_states, next_actions], dim=-1)),
                self.target_qf2(torch.cat([next_states, next_actions], dim=-1))
            )
            target_q_values = rewards + (1. - dones) * self.gamma * (target_q_values - alpha * next_log_pi) # 计算TD Target
            #print("target_q_values",target_q_values)

        # CQL Regularization
        # Ensure the generated action range is broader than data action range for effective training
        # 计算两个critic网络的loss并更新
        random_actions = torch.FloatTensor(states.shape[0], actions.shape[-1]).uniform_(-1000, 1000).to(self.device) # 这里是与SAC不同的部分，即CQL的改动
        q1_rand = self.qf1(torch.cat([states, random_actions], dim=-1))
        q2_rand = self.qf2(torch.cat([states, random_actions], dim=-1))

        cat_q1 = torch.cat([q1_rand, q1_pred, q1_pred], 1)
        cat_q2 = torch.cat([q2_rand, q2_pred, q2_pred], 1)

        min_qf1_loss = torch.logsumexp(cat_q1 / self.temperature,
                                       dim=1).mean() * self.min_q_weight * self.temperature - q1_pred.mean() * self.min_q_weight
        min_qf2_loss = torch.logsumexp(cat_q2 / self.temperature,
                                       dim=1).mean() * self.min_q_weight * self.temperature - q2_pred.mean() * self.min_q_weight

        #Expectile Regression
        diff1 = target_q_values - q1_pred
        diff2 = target_q_values - q2_pred

        qf1_loss = torch.where(diff1 > 0, self.expectile * diff1 ** 2, (1 - self.expectile) * diff1 ** 2).mean() + min_qf1_loss
        qf2_loss = torch.where(diff2 > 0, self.expectile * diff2 ** 2, (1 - self.expectile) * diff2 ** 2).mean() + min_qf2_loss

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        # Soft update target networks
        # 更新critic目标网络
        self.update_target(self.qf1, self.target_qf1)
        self.update_target(self.qf2, self.target_qf2)

        return qf1_loss.cpu().data.numpy(), qf2_loss.cpu().data.numpy(), policy_loss.cpu().data.numpy()


    def take_actions(self, states: torch.Tensor) -> np.ndarray:
        '''
        take action
        '''
        states = states.type(torch.float).to(self.device)
        actions= self.policy(states)  # Unpack the tuple to get the actions tensor
        #actions = torch.clamp(actions, min=-1, max=1)  # Updated clamping range
        actions = actions.cpu().data.numpy()
        return actions

    def forward(self, states: torch.Tensor,eval_flag: bool = True):
        if not eval_flag:
            action = self.policy(states)
        else:
            action = self.policy(states,reparameterize = False)
        return action

    def update_target(self, local_model: nn.Module, target_model: nn.Module):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_((1. - self.tau) * target_param.data + self.tau * local_param.data)

    def save_net(self, save_path: str) -> None:
        '''
        save model
        '''
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        torch.save(self.qf1.state_dict(), save_path + "/qf1" + ".pkl")
        torch.save(self.qf2.state_dict(), save_path + "/qf2" + ".pkl")
        torch.save(self.policy.state_dict(), save_path + "/policy" + ".pkl")

    def save_jit(self, save_path: str) -> None:
        '''
        save model as JIT
        '''
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        scripted_policy = torch.jit.script(self.cpu())
        scripted_policy.save(save_path + "/cql_model" + ".pth")

    def load_net(self, load_path="saved_model/fixed_initial_budget", device='cuda:0') -> None:
        '''
        load model
        '''
        self.device = device
        self.qf1.load_state_dict(torch.load(load_path + "/qf1" + ".pkl", map_location=self.device))
        self.qf2.load_state_dict(torch.load(load_path + "/qf2" + ".pkl", map_location=self.device))
        self.policy.load_state_dict(torch.load(load_path + "/policy" + ".pkl", map_location=self.device))
        self.qf1.to(self.device)
        self.qf2.to(self.device)
        self.policy.to(self.device)
        self.target_qf1.to(self.device)
        self.target_qf2.to(self.device)

if __name__ == '__main__':
    model = CQL(dim_obs=2)
    model.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(model.device)
    print("Learnable parameters: {:,}".format(model.total_params))

    a_list = [0.82091283, 0.32363808]
    b_list = [2.95450146, 2.43910137]

    step_num = 100
    batch_size = 10
    for i in range(step_num):
        states = np.random.uniform(2, 5, size=(batch_size, 2))
        next_states = np.random.uniform(2, 5, size=(batch_size, 2))
        actions = np.random.uniform(-1, 1, size=(batch_size, 1))
        rewards = np.random.uniform(0, 1, size=(batch_size, 1))
        terminals = np.zeros((batch_size, 1))
        states, next_states, actions, rewards, terminals = torch.tensor(states, dtype=torch.float), torch.tensor(
            next_states, dtype=torch.float), torch.tensor(actions, dtype=torch.float), torch.tensor(rewards,
                                                                                                    dtype=torch.float), torch.tensor(
            terminals, dtype=torch.float)

        q_loss, v_loss, a_loss = model.step(states, actions, rewards, next_states, terminals)
        print(f'step:{i} q_loss:{q_loss} v_loss:{v_loss} a_loss:{a_loss}')

    total_params = sum(p.numel() for p in model.parameters())
    print("Learnable parameters: {:,}".format(total_params))


