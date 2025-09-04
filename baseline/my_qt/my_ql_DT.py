import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0 # 确保嵌入维度 n_embd 能被注意力头的数量 n_head 整除，因为每个注意力头会处理 n_embd // n_head 维的特征。
        # 定义三个线性变换层，分别用于计算注意力机制中的 Key、Query 和 Value。
        self.key = nn.Linear(config['n_embd'], config['n_embd'])
        self.query = nn.Linear(config['n_embd'], config['n_embd'])
        self.value = nn.Linear(config['n_embd'], config['n_embd'])

        self.attn_drop = nn.Dropout(config['attn_pdrop'])# 在注意力权重上应用 dropout，防止过拟合。
        self.resid_drop = nn.Dropout(config['resid_pdrop'])# 在输出上应用 dropout。

        # 注册一个下三角矩阵（torch.tril），用于实现因果掩码（Causal Mask），防止当前时刻关注未来的输入。
        self.register_buffer("bias",
                             torch.tril(torch.ones(config['n_ctx'], config['n_ctx'])).view(1, 1, config['n_ctx'],
                                                                                           config['n_ctx']))
        # 注册一个非常大的负数（-1e4），用于掩盖无效的注意力权重。
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.proj = nn.Linear(config['n_embd'], config['n_embd'])# 定义一个线性变换层，用于将多头注意力的输出投影回原始维度。
        self.n_head = config['n_head']# 注意力头的数量。

    def forward(self, x, mask):
        B, T, C = x.size()

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
        self._attn_map = att.clone()
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

    def forward(self, inputs_embeds, attention_mask):
        x = inputs_embeds + self.attn(self.ln1(inputs_embeds), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):

    def __init__(self, state_dim, act_dim, state_mean, state_std, action_tanh=False, K=10, max_ep_len=96, scale=2000,
                 target_return=4, model_config=None):# 这里要改成dtconfig
        super(DecisionTransformer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.length_times = 3
        self.hidden_size = deconfig['hidden_size']
        self.state_mean = dtconfig['state_mean']
        self.state_std = dtconfig['state_std']
        # assert self.hidden_size == config['n_embd']
        self.max_length = dtconfig['K']
        self.max_ep_len = dtconfig['max_ep_len']

        self.state_dim = dtconfig['state_dim']
        self.act_dim = dtconfig['act_dim']
        self.scale = dtconfig['scale']
        self.target_return = dtconfig['target_return']

        self.warmup_steps = 10000
        self.weight_decay = 0.0001
        self.learning_rate = 0.0001

        # Define default block_config
        default_block_config = {
            "n_ctx": 1024,
            "n_embd": 64,
            "n_layer": 3,
            "n_head": 1,
            "n_inner": 512,
            "activation_function": "relu",
            "n_position": 1024,
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1
        }

        # If model_config is provided, override block_config with model_config values
        if dtconfig['model_config'] is None:
            block_config = default_block_config
        else:
            # Start with the default configuration
            block_config = default_block_config.copy()
            # Update defaults with values from model_config
            block_config.update(dtconfig['model_config'])

        self.transformer = nn.ModuleList([Block(block_config) for _ in range(block_config['n_layer'])])

        self.embed_timestep = nn.Embedding(self.max_ep_len, self.hidden_size)
        self.embed_return = torch.nn.Linear(1, self.hidden_size)
        self.embed_reward = torch.nn.Linear(1, self.hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)

        self.embed_ln = nn.LayerNorm(self.hidden_size)
        self.predict_state = torch.nn.Linear(self.hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(self.hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )

        self.predict_return = torch.nn.Linear(self.hidden_size, 1)
        self.predict_rewards = torch.nn.Linear(self.hidden_size, 1)

        # self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
        #                                                    lambda steps: min((steps + 1) / self.warmup_steps, 1))

        # self.init_eval()

    def forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None):
        # 将数据转移到GPU
        states, actions, rewards, returns_to_go = states.to(self.device), actions.to(self.device), rewards.to(self.device), returns_to_go.to(self.device)
        timesteps, attention_mask = timesteps.to(self.device), attention_mask.to(self.device)

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        state_embeddings = self.embed_state(states) # [256, 20, 16] -> [256, 20, 64]
        action_embeddings = self.embed_action(actions) # [256, 20 ,1] -> [256, 20 ,64]
        returns_embeddings = self.embed_return(returns_to_go) # [256, 20 ,1] -> [256, 20 ,64]
        rewards_embeddings = self.embed_reward(rewards) # [256, 20 ,1] -> [256, 20 ,64]
        time_embeddings = self.embed_timestep(timesteps) # [256, 20] -> [256, 20 ,64]

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        rewards_embeddings = rewards_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1 # 在第1维（dim=1）上将这三个张量堆叠，得到一个四维张量，形状为 [batch_size, 3, seq_length, hidden_size]。
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self.hidden_size) # 将 [batch_size, 3, seq_length, hidden_size] 转换为 [batch_size, seq_length, 3, hidden_size]。然后展平为[batch_size, 3 * seq_length, hidden_size]。[256, 60, 64]
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            ([attention_mask for _ in range(self.length_times)]), dim=1 # 将 attention_mask 重复 self.length_times 次。在第1维（dim=1）上堆叠
        ).permute(0, 2, 1).reshape(batch_size, self.length_times * seq_length).to(stacked_inputs.dtype) # 重新排列并展平[256, 60]

        x = stacked_inputs
        for block in self.transformer:
            x = block(x, stacked_attention_mask)

        x = x.reshape(batch_size, seq_length, self.length_times, self.hidden_size).permute(0, 2, 1, 3) # [256, 60, 64] -> [256, 3, 20, 64]

        return_preds = self.predict_return(x[:, 2]) # [256, 20, 64] -> [256, 20, 1]
        state_preds = self.predict_state(x[:, 2])   # [256, 20, 64] -> [256, 20, 16]
        action_preds = self.predict_action(x[:, 1]) # [256, 20, 64] -> [256, 20, 1]
        return state_preds, action_preds, return_preds, None

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


class QlModel(nn.Module):
    '''
    Use DecisionTransformer and Critic
    '''
    def __init__(
            self,
            state_dim,
            act_dim,
            state_mean,
            state_std,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            sar=False,
            scale=1.,
            rtg_no_q=False,
            infer_no_q=False,
            target_return = 4,
            dtconfig = None,
            **kwargs
    ):
        super(QlModel, self).__init__()
        self.hidden_size = hidden_size
        self.sar = sar
        self.scale = scale
        self.rtg_no_q = rtg_no_q
        self.infer_no_q = infer_no_q
        self.target_return = target_return
        self.state_mean = state_mean
        self.state_std = state_std

        self.transformer = DecisionTransformer(dtconfig)

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

