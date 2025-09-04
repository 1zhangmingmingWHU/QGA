import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
                 target_return=4, model_config=None):
        super(DecisionTransformer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.length_times = 3
        self.hidden_size = 64
        self.state_mean = state_mean
        self.state_std = state_std
        # assert self.hidden_size == config['n_embd']
        self.max_length = K
        self.max_ep_len = max_ep_len

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.scale = scale
        self.target_return = target_return

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
        if model_config is None:
            block_config = default_block_config
        else:
            # Start with the default configuration
            block_config = default_block_config.copy()
            # Update defaults with values from model_config
            block_config.update(model_config)

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

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                           lambda steps: min((steps + 1) / self.warmup_steps, 1))

        self.init_eval()

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

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs):
        # 将数据转移到GPU
        # 传入的state的形状是[seq_len, 16]
        states, actions, rewards, returns_to_go, timesteps = states.to(self.device), actions.to(self.device), rewards.to(self.device), returns_to_go.to(self.device), timesteps.to(self.device)
        
        # we don't care about the past rewards in this model
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

        _, action_preds, return_preds, reward_preds = self.forward(
            states, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0, -1]

    def step(self, states, actions, rewards, dones, rtg, timesteps, attention_mask):
        rewards_target, action_target, rtg_target = torch.clone(rewards), torch.clone(actions), torch.clone(rtg) # rewards_target 和 rtg_target 在后续代码中未被使用

        state_preds, action_preds, return_preds, reward_preds = self.forward( # 这里reward_preds是none
            states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask, # rtg[:, :-1]：使用 rtg 的前 seq_length-1 个时间步作为输入
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0] # [256, 20 ,1] -> [4013, 1]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0] # [256, 20 ,1] -> [4013, 1]

        loss = torch.mean((action_preds - action_target) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), .25)
        self.optimizer.step()

        return loss.detach().cpu().item()

    def take_actions(self, state, target_return=None, pre_reward=None):# target_return（可选）：目标回报，用于指导策略生成动作。pre_reward（可选）：前一步的奖励（仅在非初始步时传入）。
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
            (self.eval_states.to(dtype=torch.float32) - self.state_mean) / self.state_std, # [2, 16]
            self.eval_actions.to(dtype=torch.float32),                                     # [2, 1] 
            self.eval_rewards.to(dtype=torch.float32),
            self.eval_target_return.to(dtype=torch.float32), # 就是rtg
            self.eval_timesteps.to(dtype=torch.long)
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


    def save_net(self, save_path, block_config):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Save the model's state_dict
        file_path = os.path.join(save_path, "dt.pt")
        torch.save(self.state_dict(), file_path)
        # Save the block configuration as a JSON file
        config_path = os.path.join(save_path, "block_config.json")
        with open(config_path, 'w') as f:
            json.dump(block_config, f, indent=4)

    def save_jit(self, save_path, block_config):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # Save the JIT compiled model
        jit_model = torch.jit.script(self.cpu())
        torch.jit.save(jit_model, os.path.join(save_path, 'dt_model.pth'))
        # Save the block configuration as a JSON file
        config_path = os.path.join(save_path, "block_config.json")
        with open(config_path, 'w') as f:
            json.dump(block_config, f, indent=4)


    # 这里写得有点逆天。device和self.device不是一个东西。实际上用的是传进来的device，这里又print的是self.device。
    def load_net(self, load_path="saved_model/DTtest", device=None):
        file_path = load_path
        if device is None:
            device = self.device
        self.load_state_dict(torch.load(file_path, map_location=device))
        self.to(device)
        print(f"Model loaded from {self.device}.")
