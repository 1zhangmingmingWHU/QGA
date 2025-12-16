import torch
from torch.utils.data import Dataset
import ast
import numpy as np
import random
import pandas as pd


def getScore_nips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward, penalty

class EpisodeReplayBuffer(Dataset):
    def __init__(self, state_dim, act_dim, data_path='data/autoBidding_aigb_track_data_trajectory_data.csv', max_ep_len=48, scale=50, K=10, sparse_data=False):
        self.device = "cpu"
        super(EpisodeReplayBuffer, self).__init__()
        self.max_ep_len = max_ep_len
        self.scale = scale

        self.state_dim = state_dim
        self.act_dim = act_dim

        print(f'loading data from {data_path}...')
        training_data = pd.read_csv(data_path)

        # this could be very slow
        def safe_literal_eval(val):
            if pd.isna(val):
                return val
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                print(ValueError)
                return val

        training_data["state"] = training_data["state"].apply(safe_literal_eval)
        training_data["next_state"] = training_data["next_state"].apply(safe_literal_eval)
        training_data["next_state"] = training_data["state"].shift(-1)
        training_data.at[training_data.index[-1], 'next_state'] = training_data.at[0, 'state']
        self.trajectories = training_data

        self.states, self.rewards, self.actions, self.returns, self.traj_lens, self.dones = [], [], [], [], [], []
        self.budget, self.CPA_constrain = [], []
        self.cost_ts = []

        # debug = 0
        state = []
        reward = []
        action = []
        dones = []
        cost_ts = []
        for index, row in self.trajectories.iterrows():
            state.append(row["state"])
            reward.append(row['reward'])
            action.append(row["action"])
            dones.append(row["done"])
            cost_t = (row['realAllCost'] - (1-row["state"][1]) * row['budget']) if row['done'] else (row["state"][1] - row['next_state'][1]) * row['budget']
            cost_ts.append(cost_t)
            training_data.loc[index, "cost"] = cost_t

            if row["done"]:
                if len(state) != 1:
                    self.states.append(np.array(state))
                    self.rewards.append(np.expand_dims(np.array(reward), axis=1))
                    self.actions.append(np.expand_dims(np.array(action), axis=1))
                    self.returns.append(sum(reward))
                    self.traj_lens.append(len(state))
                    self.dones.append(np.array(dones))
                    self.cost_ts.append(np.array(cost_ts))
                    self.budget.append(row['budget'])
                    self.CPA_constrain.append(row['CPAConstraint'])

                state = []
                reward = []
                action = []
                dones = []
                cost_ts = []

                # debug += 1
                # if debug >=1024 and debug_mode:
                #     break

        self.traj_lens, self.returns = np.array(self.traj_lens), np.array(self.returns)

        tmp_states = np.concatenate(self.states, axis=0)
        self.state_mean, self.state_std = np.mean(tmp_states, axis=0), np.std(tmp_states, axis=0)
        self.state_max, self.state_min = np.max(tmp_states, axis=0), np.min(tmp_states, axis=0)
        self.returns_max, self.returns_min = np.max(self.returns, axis=0), 0

        self.trajectories = []

        for i in range(len(self.states)):
            self.trajectories.append(
                {"observations": self.states[i], "actions": self.actions[i], "rewards": self.rewards[i],
                "dones": self.dones[i],
                "cost_ts": self.cost_ts[i],
                "budget": self.budget[i],
                "cpa_constrain": self.CPA_constrain[i],
                })

        # save the preprocessed data and concat them later to a single file
        #     np.save(f'data/preprocessed_trajectory_data_{data_path[-7:-4]}.npy', self.trajectories)

        self.K = K  # horizon of a trajectory, i.e., how many historical steps can a DT see
        self.pct_traj = 1.

        num_timesteps = sum(self.traj_lens)
        num_timesteps = max(int(self.pct_traj * num_timesteps), 1)
        sorted_inds = np.argsort(self.returns)  # lowest to highest
        num_trajectories = 1
        timesteps = self.traj_lens[sorted_inds[-1]]
        ind = len(self.trajectories) - 2
        while ind >= 0 and timesteps + self.traj_lens[sorted_inds[ind]] <= num_timesteps:
            timesteps += self.traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        self.sorted_inds = sorted_inds[-num_trajectories:]

        self.p_sample = self.traj_lens[self.sorted_inds] / sum(self.traj_lens[self.sorted_inds])


    def __getitem__(self, index):
        traj = self.trajectories[int(self.sorted_inds[index])]
        start_t = random.randint(0, traj['rewards'].shape[0] - 1)

        s = traj['observations'][start_t: start_t + self.K]
        a = traj['actions'][start_t: start_t + self.K]
        r = traj['rewards'][start_t: start_t + self.K].reshape(-1, 1)
        c = traj["cost_ts"][start_t: start_t + self.K].reshape(-1, 1)
        if 'terminals' in traj:
            d = traj['terminals'][start_t: start_t + self.K]
        else:
            d = traj['dones'][start_t: start_t + self.K]
        timesteps = np.arange(start_t, start_t + s.shape[0])
        timesteps[timesteps >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff

        rtg = self.discount_cumsum(traj['rewards'][start_t:], gamma=1.)[:s.shape[0] + 1].reshape(-1, 1)
        if rtg.shape[0] <= s.shape[0]:
            rtg = np.concatenate([rtg, np.zeros((1, 1))], axis=0)

        # implement the ctg = min{(CPA_constraint / C_{t:T})^beta, 1}
        cost_t_T = self.discount_cumsum(traj['cost_ts'][start_t:], gamma=1.)[:s.shape[0] + 1].reshape(-1, 1)
        if cost_t_T.shape[0] <= s.shape[0]:
            cost_t_T = np.concatenate([cost_t_T, np.zeros((1, 1))], axis=0)

        C_t_T = cost_t_T / (rtg + 1e-5)

        beta = 2
        ctg = np.ones_like(C_t_T)
        for ci, ctg_i in enumerate(C_t_T):
            if ctg_i > traj["cpa_constrain"]:
                coef = traj["cpa_constrain"] / (ctg_i + 1e-10)
                ctg[ci] = pow(coef, beta)

        # score_{t:T} = score_{0:T} - score_{0:t-1}
        score_0_T, _ = getScore_nips(sum(traj['rewards']), sum(traj["cost_ts"])/(sum(traj['rewards'])+1e-5), traj["cpa_constrain"])
        score_t_T = np.zeros_like(rtg)
        for t in range(start_t, start_t + s.shape[0]+1):
            score_0_tm1, _ = getScore_nips(sum(traj['rewards'][:t]), sum(traj["cost_ts"][:t])/(sum(traj['rewards'][:t])+1e-5), traj["cpa_constrain"])
            score_t_T[t-start_t] = score_0_T - score_0_tm1

        tlen = s.shape[0]
        s = np.concatenate([np.zeros((self.K - tlen, self.state_dim)), s], axis=0)
        s = (s - self.state_mean) / self.state_std
        a = np.concatenate([np.ones((self.K - tlen, self.act_dim)) * -10., a], axis=0)
        r = np.concatenate([np.zeros((self.K - tlen, 1)), r], axis=0)
        r = r / self.scale

        # cost for every step t and we normalize it to 0～1 by dividing its budget
        c = np.concatenate([np.zeros((self.K - tlen, 1)), c], axis=0)
        c = c / traj["budget"]

        d = np.concatenate([np.ones((self.K - tlen)) * 2, d], axis=0)
        rtg = np.concatenate([np.zeros((self.K - tlen, 1)), rtg], axis=0) / self.scale
        ctg = np.concatenate([np.zeros((self.K - tlen, 1)), ctg], axis=0)    # ctg is already in [0,1], no need to scale
        score_t_T = np.concatenate([np.zeros((self.K - tlen, 1)), score_t_T], axis=0) / self.scale

        timesteps = np.concatenate([np.zeros((self.K - tlen)), timesteps], axis=0)
        mask = np.concatenate([np.zeros((self.K - tlen)), np.ones((tlen))], axis=0)

        s = torch.from_numpy(s).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(a).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(r).to(dtype=torch.float32, device=self.device)
        c = torch.from_numpy(c).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(d).to(dtype=torch.long, device=self.device)
        rtg = torch.from_numpy(rtg).to(dtype=torch.float32, device=self.device)
        ctg = torch.from_numpy(ctg).to(dtype=torch.float32, device=self.device)
        score_t_T = torch.from_numpy(score_t_T).to(dtype=torch.float32, device=self.device)

        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(mask).to(device=self.device)
        
        return s, a, r, d, rtg, timesteps, mask, ctg, score_t_T, c

    def discount_cumsum(self, x, gamma=1.):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum
