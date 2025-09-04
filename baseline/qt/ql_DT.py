import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

from bidding_train_env.baseline.qt.model import TrajectoryModel
from bidding_train_env.baseline.qt.trajectory_gpt2 import GPT2Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            state_mean,
            state_std,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=False,
            sar=False,
            scale=1.,
            rtg_no_q=False,
            infer_no_q=False,
            target_return = 4,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )
        self.config = config
        self.sar = sar
        self.scale = scale
        self.rtg_no_q = rtg_no_q
        self.infer_no_q = infer_no_q
        self.target_return = target_return
        self.state_mean = state_mean
        self.state_std = state_std

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_rewards = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_rewards = torch.nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards=None, targets=None, returns_to_go=None, timesteps=None, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        reward_embeddings = self.embed_rewards(rewards / self.scale)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        reward_embeddings = reward_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        if self.sar:
            stacked_inputs = torch.stack(
                (state_embeddings, action_embeddings, reward_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        else:
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        if self.sar:
            action_preds = self.predict_action(x[:, 0])
            rewards_preds = self.predict_rewards(x[:, 1])
            state_preds = self.predict_state(x[:, 2])
        else:
            action_preds = self.predict_action(x[:, 1])
            state_preds = self.predict_state(x[:, 2])
            rewards_preds = None


        return state_preds, action_preds, rewards_preds

    def get_action(self, critic, states, actions, rewards=None, returns_to_go=None, timesteps=None, **kwargs): # 这里对应论文中使用Q值模块进行推理的部分
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim).repeat_interleave(repeats=50, dim=0) # 将输入序列 reshape 为 (50, seq_len, dim) 的形式（seq_len 是序列长度，dim 是维度）。在 batch 维度（dim=0）上重复 50 次，生成 50 个相同的序列（用于后续的随机采样或多样性生成）。
        actions = actions.reshape(1, -1, self.act_dim).repeat_interleave(repeats=50, dim=0) # [50, seq_len, 1]
        rewards = rewards.reshape(1, -1, 1).repeat_interleave(repeats=50, dim=0) # [50, seq_len, 1]
        timesteps = timesteps.reshape(1, -1).repeat_interleave(repeats=50, dim=0) # [50,seq_len]

        bs = returns_to_go.shape[0] # 数组长度 bs永远都是1，是不是写错了
        returns_to_go = returns_to_go.reshape(bs, -1, 1).repeat_interleave(repeats=50 // bs, dim=0) # 将 returns_to_go reshape 为 (bs, seq_len, 1) 的形式，再在 batch 维度上重复 50 // bs 次。
        returns_to_go = torch.cat([returns_to_go, torch.randn((50-returns_to_go.shape[0], returns_to_go.shape[1], 1), device=returns_to_go.device)], dim=0) # 用随机噪声填充剩余部分（确保总 batch 大小为 50）。[50, seq_len, 1]
            

        if self.max_length is not None: # 如果序列长度超过 self.max_length，则只保留最近的 max_length 步。
            states = states[:,-self.max_length:] # 形状都没变
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            rewards = rewards[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # padding 如果序列长度不足 max_length，则使用零填充。
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])]) # [20]
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1).repeat_interleave(repeats=50, dim=0) #[50,20]
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32) # [50,20,16]
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
            rewards = torch.cat(
                [torch.zeros((rewards.shape[0], self.max_length-rewards.shape[1], 1), device=rewards.device), rewards],
                dim=1
            ).to(dtype=torch.float32)

            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length-actions.shape[1], self.act_dim), device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
        else:
            attention_mask = None

        returns_to_go[bs:, -1] = returns_to_go[bs:, -1] + torch.randn_like(returns_to_go[bs:, -1]) * 0.1 # 对目标回报的最后一个时间步添加高斯噪声，增加多样性。 returns_to_go的形状是[50,20,1]
        if not self.rtg_no_q: # 如果启用 Q 值引导（self.rtg_no_q=False），则用 critic 的 Q 值更新最后一个目标回报。
            returns_to_go[-1, -1] = critic.q_min(states[-1:, -2], actions[-1:, -2]).flatten() - rewards[-1, -2] / self.scale
        # 调用模型的 forward 方法，生成预测动作 action_preds 和预测回报 return_preds。
        _, action_preds, return_preds = self.forward(states, actions, rewards, None, returns_to_go=returns_to_go, timesteps=timesteps, attention_mask=attention_mask, **kwargs)
    
        
        state_rpt = states[:, -1, :] # [50,16]
        action_preds = action_preds[:, -1, :] # [50,1]

        q_value = critic.q_min(state_rpt, action_preds).flatten() # 使用 critic 计算状态和预测动作的 Q 值。
        idx = torch.multinomial(F.softmax(q_value, dim=-1), 1) # 根据 Q 值的 softmax 分布进行采样，选择动作。

        if not self.infer_no_q: # 如果启用 Q 值引导（self.infer_no_q=False），返回采样动作。
            return action_preds[idx].squeeze()
        else: # 否则，返回第一个预测动作。
            return action_preds[0]

    def get_action_simple(self, critic, states, actions, rewards=None, returns_to_go=None, timesteps=None, **kwargs): # 不使用Q值模块进行推理
        # states, actions, rewards, returns_to_go, timesteps = states.to(self.device), actions.to(self.device), rewards.to(self.device), returns_to_go.to(self.device), timesteps.to(self.device)
        
        states = states.reshape(1, -1, self.state_dim) # [1,seq_len, 16]
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        rewards = rewards.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1) # [1, seq_len]

        if self.max_length is not None:
            states = states[:, -self.max_length:] # 这里都没变
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            rewards = rewards[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            attention_mask = torch.cat([torch.zeros(self.max_length - states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length - states.shape[1], self.state_dim),
                             device=states.device), states],
                dim=1).to(dtype=torch.float32) # 填充到最大长度 [1, max_seq_len, 16]
            actions = torch.cat( 
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat( # [1, max_seq_len, 1]
                [torch.zeros((returns_to_go.shape[0], self.max_length - returns_to_go.shape[1], 1),
                             device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            rewards = torch.cat( # [1, max_seq_len, 1]
                [torch.zeros((rewards.shape[0], self.max_length - rewards.shape[1], 1), device=rewards.device),
                 rewards],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat( # [1, max_seq_len]
                [torch.zeros((timesteps.shape[0], self.max_length - timesteps.shape[1]), device=timesteps.device),
                 timesteps],
                dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        # 调用模型的 forward 方法，生成预测动作 action_preds 和预测回报 return_preds。
        _, action_preds, return_preds = self.forward(states, actions, rewards, None, returns_to_go=returns_to_go, timesteps=timesteps, attention_mask=attention_mask, **kwargs)
    
        return action_preds[0, -1]



    def take_actions(self, critic, state, target_return=None, pre_reward=None, budget=100, cpa=2):
        self.eval()
        if self.eval_states is None:
            self.eval_states = torch.from_numpy(state).reshape(1, self.state_dim) # 将 state 转换为 PyTorch 张量，并 reshape 为 (1, state_dim)。
            ep_return = target_return if target_return is not None else self.target_return # 如果未提供 target_return，则使用默认的 self.target_return。
            self.eval_target_return = torch.tensor(ep_return, dtype=torch.float32).reshape(1, 1) # eval_target_return 初始化为目标回报的张量，用于后续的序列生成。
        else:
            assert pre_reward is not None # 确保前一步的奖励已提供
            cur_state = torch.from_numpy(state).reshape(1, self.state_dim) # 将传入的 state 转换为 PyTorch 张量，并 reshape 为 (1, state_dim)。
            self.eval_states = torch.cat([self.eval_states, cur_state], dim=0) # 将当前状态 cur_state 拼接到历史状态序列 eval_states 中
            self.eval_rewards[-1] = pre_reward # 用 pre_reward 更新最近一步的奖励（覆盖之前初始化的零值）。
            pred_return = self.eval_target_return[0, -1] - (pre_reward / self.scale) # 计算 pred_return = 上一步目标回报 - (pre_reward / scale)
            self.eval_target_return = torch.cat([self.eval_target_return, pred_return.reshape(1, 1)], dim=1) # 将新的目标回报拼接到 eval_target_return 序列中。
            self.eval_timesteps = torch.cat(
                [self.eval_timesteps, torch.ones((1, 1), dtype=torch.long) * self.eval_timesteps[:, -1] + 1], dim=1) # 时间步数 eval_timesteps 递增 1。
        self.eval_actions = torch.cat([self.eval_actions, torch.zeros(1, self.act_dim)], dim=0) # 在 eval_actions 中追加一个零动作（占位，后续会被覆盖）。
        self.eval_rewards = torch.cat([self.eval_rewards, torch.zeros(1)]) # 在 eval_rewards 中追加一个零奖励（占位，下一步会被 pre_reward 更新）。

        action = self.get_action( # 生成动作
            critic,
            ((self.eval_states.to(dtype=torch.float32) - self.state_mean) / self.state_std).to(device), # [2, 16]
            (self.eval_actions.to(dtype=torch.float32)).to(device),                                     # [2, 1] 
            (self.eval_rewards.to(dtype=torch.float32)).to(device),
            (self.eval_target_return.to(dtype=torch.float32)).to(device), # 就是rtg
            (self.eval_timesteps.to(dtype=torch.long)).to(device)
        )
        self.eval_actions[-1] = action # 用新生成的动作覆盖占位零。
        action = action.detach().cpu().numpy() # 将动作转为 numpy 格式并返回。

        return action

    def init_eval(self):
        self.eval_states = None
        self.eval_actions = torch.zeros((0, self.act_dim), dtype=torch.float32)
        self.eval_rewards = torch.zeros(0, dtype=torch.float32)

        self.eval_target_return = None
        self.eval_timesteps = torch.tensor(0, dtype=torch.long).reshape(1, 1)

        self.eval_episode_return, self.eval_episode_length = 0, 0