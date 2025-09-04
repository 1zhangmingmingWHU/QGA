import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 自定义 Transformer Block（from dt_baselines.py）
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        self.key = nn.Linear(config['n_embd'], config['n_embd'])
        self.query = nn.Linear(config['n_embd'], config['n_embd'])
        self.value = nn.Linear(config['n_embd'], config['n_embd'])
        self.attn_drop = nn.Dropout(config['attn_pdrop'])
        self.resid_drop = nn.Dropout(config['resid_pdrop'])
        self.register_buffer("bias", torch.tril(torch.ones(config['n_ctx'], config['n_ctx'])).view(1, 1, config['n_ctx'], config['n_ctx']))
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.n_head = config['n_head']

    def forward(self, x, mask): 
        B, T, C = x.size() # T=seq*num_item, C=emb_dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        mask = mask.view(B, -1)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.where(self.bias[:, :, :T, :T].bool(), att, self.masked_bias.to(att.dtype))
        att = att + mask
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config['n_embd'], config['n_inner']),
            nn.GELU(),
            nn.Linear(config['n_inner'], config['n_embd']),
            nn.Dropout(config['resid_pdrop']),
        )

    def forward(self, inputs_embeds, attention_mask): # batch*(seq*3)*dim, batch*(seq*3)
        x = inputs_embeds + self.attn(self.ln1(inputs_embeds), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class Critic(nn.Module):  # 原样复制
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

class DecisionTransformer(nn.Module):
    """
    GPT2-free Decision Transformer. Transformer backbone is custom, see dt_baselines.Block.
    Supports rtg_no_q, infer_no_q, target_return, max_length, standard scaling, and Q-guided action selection.
    """
    def __init__(
        self,
        state_dim,
        act_dim,
        state_mean,
        state_std,
        hidden_size=512,
        max_length=20,
        max_ep_len=4096,
        action_tanh=False,
        sar=False,
        scale=1.,
        rtg_no_q=False,
        infer_no_q=False,
        target_return=4,
        **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.max_ep_len = max_ep_len
        self.sar = sar
        self.scale = scale
        self.rtg_no_q = rtg_no_q
        self.infer_no_q = infer_no_q
        self.target_return = target_return
        self.state_mean = torch.tensor(state_mean, dtype=torch.float32).to(device)
        self.state_std = torch.tensor(state_std, dtype=torch.float32).to(device)
        self.hidden_size = hidden_size
        self.length_times = 3  # (R, s, a) / (s, a, r)
        # transformer config
        block_config = {
            "n_ctx": self.max_length * self.length_times,
            "n_embd": self.hidden_size,
            "n_layer": 6,
            "n_head": 8,
            "n_inner": self.hidden_size,
            "activation_function": "gelu",
            "n_position": self.max_length * self.length_times,
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1
        }
        # backbone
        self.transformer = nn.ModuleList([Block(block_config) for _ in range(block_config['n_layer'])])
        self.embed_timestep = nn.Embedding(self.max_ep_len, self.hidden_size)
        self.embed_return = torch.nn.Linear(1, self.hidden_size)
        self.embed_rewards = torch.nn.Linear(1, self.hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)
        self.embed_ln = nn.LayerNorm(self.hidden_size)
        self.predict_state = torch.nn.Linear(self.hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_rewards = torch.nn.Linear(self.hidden_size, 1)
        # for rollout
        self.init_eval()

    def forward(self, states, actions, rewards=None, targets=None, returns_to_go=None, timesteps=None, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        reward_embeddings = self.embed_rewards(rewards / self.scale)
        time_embeddings = self.embed_timestep(timesteps)

        # sum time embedding
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        reward_embeddings = reward_embeddings + time_embeddings

        if self.sar:
            stacked_inputs = torch.stack(
                (state_embeddings, action_embeddings, reward_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        else:
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # pad attention mask
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length).to(stacked_inputs.dtype)

        # forward n transformer blocks
        x = stacked_inputs
        for block in self.transformer:
            x = block(x, stacked_attention_mask)
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        if self.sar:
            action_preds = self.predict_action(x[:, 0])
            rewards_preds = self.predict_rewards(x[:, 1])
            state_preds = self.predict_state(x[:, 2])
        else:
            action_preds = self.predict_action(x[:, 1])
            state_preds = self.predict_state(x[:, 2])
            rewards_preds = None

        return state_preds, action_preds, rewards_preds

    # ---------------------- 推理接口与原文件一致 ----------------------

    def get_action(self, critic, states, actions, rewards=None, returns_to_go=None, timesteps=None, **kwargs):
        # Q-guided, 生成50组候选，采样最大Q
        states = states.reshape(1, -1, self.state_dim).repeat_interleave(repeats=50, dim=0)
        actions = actions.reshape(1, -1, self.act_dim).repeat_interleave(repeats=50, dim=0)
        rewards = rewards.reshape(1, -1, 1).repeat_interleave(repeats=50, dim=0)
        timesteps = timesteps.reshape(1, -1).repeat_interleave(repeats=50, dim=0)
        bs = returns_to_go.shape[0]
        returns_to_go = returns_to_go.reshape(bs, -1, 1).repeat_interleave(repeats=50 // bs, dim=0)
        returns_to_go = torch.cat([
            returns_to_go,
            torch.randn((50-returns_to_go.shape[0], returns_to_go.shape[1], 1), device=returns_to_go.device)
        ], dim=0)
        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            rewards = rewards[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]
            attention_mask = torch.cat([
                torch.zeros(self.max_length-states.shape[1], device=states.device),
                torch.ones(states.shape[1], device=states.device)
            ])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1).repeat_interleave(repeats=50, dim=0)
            # pad
            states = torch.cat([
                torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device),
                states
            ], dim=1).to(dtype=torch.float32)
            actions = torch.cat([
                torch.zeros((actions.shape[0], self.max_length-actions.shape[1], self.act_dim), device=actions.device),
                actions
            ], dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat([
                torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device),
                returns_to_go
            ], dim=1).to(dtype=torch.float32)
            rewards = torch.cat([
                torch.zeros((rewards.shape[0], self.max_length-rewards.shape[1], 1), device=rewards.device),
                rewards
            ], dim=1).to(dtype=torch.float32)
            timesteps = torch.cat([
                torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device),
                timesteps
            ], dim=1).to(dtype=torch.long)
        else:
            attention_mask = None
        returns_to_go[bs:, -1] = returns_to_go[bs:, -1] + torch.randn_like(returns_to_go[bs:, -1]) * 0.1
        if not self.rtg_no_q:
            returns_to_go[-1, -1] = critic.q_min(states[-1:, -2], actions[-1:, -2]).flatten() - rewards[-1, -2] / self.scale
        _, action_preds, _ = self.forward(states, actions, rewards, None, returns_to_go, timesteps, attention_mask)
        state_rpt = states[:, -1, :]
        action_preds = action_preds[:, -1, :]
        q_value = critic.q_min(state_rpt, action_preds).flatten()
        idx = torch.multinomial(F.softmax(q_value, dim=-1), 1)
        if not self.infer_no_q:
            return action_preds[idx].squeeze()
        else:
            return action_preds[0]

    def get_action_simple(self, critic, states, actions, rewards=None, returns_to_go=None, timesteps=None, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        rewards = rewards.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            rewards = rewards[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]
            attention_mask = torch.cat([
                torch.zeros(self.max_length-states.shape[1], device=states.device),
                torch.ones(states.shape[1], device=states.device)
            ])
            attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
            states = torch.cat([
                torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device),
                states
            ], dim=1).to(dtype=torch.float32)
            actions = torch.cat([
                torch.zeros((actions.shape[0], self.max_length-actions.shape[1], self.act_dim), device=actions.device),
                actions
            ], dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat([
                torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device),
                returns_to_go
            ], dim=1).to(dtype=torch.float32)
            rewards = torch.cat([
                torch.zeros((rewards.shape[0], self.max_length-rewards.shape[1], 1), device=rewards.device),
                rewards
            ], dim=1).to(dtype=torch.float32)
            timesteps = torch.cat([
                torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device),
                timesteps
            ], dim=1).to(dtype=torch.long)
        else:
            attention_mask = None
        _, action_preds, _ = self.forward(states, actions, rewards, None, returns_to_go, timesteps, attention_mask)
        return action_preds[0, -1]

    # ------------ rollout/inference流程，照ql_DT.py模式一致 ------------

    def take_actions(self, critic, state, target_return=None, pre_reward=None, budget=100, cpa=2):
        self.eval()
        if self.eval_states is None:
            self.eval_states = torch.from_numpy(state).reshape(1, self.state_dim)
            ep_return = target_return if target_return is not None else self.target_return
            self.eval_target_return = torch.tensor(ep_return, dtype=torch.float32).reshape(1, 1)
        else:
            assert pre_reward is not None
            cur_state = torch.from_numpy(state).reshape(1, self.state_dim)
            cur_state = cur_state.to(device)
            self.eval_states = torch.cat([self.eval_states, cur_state], dim=0)
            self.eval_rewards[-1] = pre_reward
            pred_return = self.eval_target_return[0, -1] - (pre_reward / self.scale)
            self.eval_target_return = torch.cat([self.eval_target_return, pred_return.reshape(1, 1)], dim=1)
            self.eval_timesteps = torch.cat([
                self.eval_timesteps, torch.ones((1, 1), dtype=torch.long) * self.eval_timesteps[:, -1] + 1
            ], dim=1)
        self.eval_actions = torch.cat([self.eval_actions, torch.zeros(1, self.act_dim)], dim=0)
        self.eval_rewards = torch.cat([self.eval_rewards, torch.zeros(1)], dim=0)

        self.eval_states = self.eval_states.to(device)

        action = self.get_action(
            critic,
            ((self.eval_states.to(dtype=torch.float32) - self.state_mean) / self.state_std).to(device),
            (self.eval_actions.to(dtype=torch.float32)).to(device),
            (self.eval_rewards.to(dtype=torch.float32)).to(device),
            (self.eval_target_return.to(dtype=torch.float32)).to(device),
            (self.eval_timesteps.to(dtype=torch.long)).to(device)
        )
        self.eval_actions[-1] = action
        action = action.detach().cpu().numpy()
        return action

    def init_eval(self):
        self.eval_states = None
        self.eval_actions = torch.zeros((0, self.act_dim), dtype=torch.float32)
        self.eval_rewards = torch.zeros(0, dtype=torch.float32)
        self.eval_target_return = None
        self.eval_timesteps = torch.tensor(0, dtype=torch.long).reshape(1, 1)
        self.eval_episode_return, self.eval_episode_length = 0, 0

# --- end file ---
