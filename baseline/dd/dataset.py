from torch.utils.data import Dataset
import pandas as pd
import ast
import numpy as np
import torch
import os


class aigb_dataset(Dataset):
    def __init__(self, args, train_data_path, step_len, **kwargs) -> None:
        super().__init__()
        
        # 检查是否存在预处理数据
        processed_data_dir = train_data_path
        if os.path.exists(processed_data_dir):
            print("加载预处理数据...")
            self.states = np.load(os.path.join(processed_data_dir, "states.npy"))
            self.actions = np.load(os.path.join(processed_data_dir, "actions.npy"))
            self.rewards = np.load(os.path.join(processed_data_dir, "rewards.npy"))
            self.terminals = np.load(os.path.join(processed_data_dir, "terminals.npy"))
        else:
            print("未找到预处理数据, 退出")
            exit(0)
        self.step_len = 48    # 48
        self.o_dim = self.states.shape[1]
        self.a_dim = self.actions.shape[1]
        self.target_len = 48
        self.random_crop = True

        # 分割序列索引映射
        # 每个序列的开头
        self.candidate_pos = (self.terminals == 0).nonzero()[0]
        self.candidate_pos += 1
        self.candidate_pos = [0] + self.candidate_pos.tolist()[:-1]
        # 后面再加上序列的结尾
        self.candidate_pos = self.candidate_pos + [self.states.shape[0]]

    def crop_sequence(self, state, action, reward):
        """序列截断函数"""
        len_state = self.step_len 
        if len_state > self.target_len:
            if self.random_crop:
                # 随机截断
                start_idx = np.random.randint(0, len_state - self.target_len)
            else:
                # 固定从开始截断
                start_idx = 0
            
            state = state[start_idx:start_idx + self.target_len]
            action = action[start_idx:start_idx + self.target_len]
            reward = reward[start_idx:start_idx + self.target_len]
            
        return state, action, reward

    def __len__(self):
        return len(self.candidate_pos) - 1

    def __getitem__(self, index):
        # 获取序列
        state = self.states[self.candidate_pos[index]:self.candidate_pos[index + 1], :]
        action = self.actions[self.candidate_pos[index]:self.candidate_pos[index + 1], :]
        reward = self.rewards[self.candidate_pos[index]:self.candidate_pos[index + 1], :]

        len_state = len(state)
        reward = np.squeeze(reward)
        
        # 进行padding,填充下方，满足总长度为 self.step_len
        state_padded = np.zeros((self.step_len, self.o_dim), dtype=np.float32)
        action_padded = np.zeros((self.step_len, self.a_dim), dtype=np.float32)
        reward_padded = np.zeros((self.step_len, 1), dtype=np.float32)
        
        state_padded[:len_state] = state
        action_padded[:len_state] = action
        reward_padded[:len_state, 0] = reward  # 确保reward是2D数组

        state, action, reward = self.crop_sequence(state_padded, action_padded, reward_padded)
        
        def discount_cumsum(x, gamma=1.):
            discount_cumsum = np.zeros_like(x)
            discount_cumsum[-1] = x[-1]
            for t in reversed(range(x.shape[0] - 1)):
                discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
            return discount_cumsum

        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)      # 如果action需要long改成long
        rtg = discount_cumsum(reward, gamma=1.)
        returns = rtg[0]
        returns = 1 / (1 + np.exp(-returns))  # sigmoid
        returns = torch.tensor(np.clip(returns, None, 1.0), dtype=torch.float32).reshape(1)
        masks = torch.zeros(self.step_len, dtype=torch.bool)
        masks[:len_state] = True

        return state, action, returns, masks
