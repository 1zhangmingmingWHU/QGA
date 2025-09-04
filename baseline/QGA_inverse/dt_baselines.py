import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math

def getScore_nips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        self.key = nn.Linear(config['n_embd'], config['n_embd'])
        self.query = nn.Linear(config['n_embd'], config['n_embd'])
        self.value = nn.Linear(config['n_embd'], config['n_embd'])

        self.attn_drop = nn.Dropout(config['attn_pdrop'])
        self.resid_drop = nn.Dropout(config['resid_pdrop'])

        # 1*1*n_ctx*n_ctx
        self.register_buffer("bias",
                             torch.tril(torch.ones(config['n_ctx'], config['n_ctx'])).view(1, 1, config['n_ctx'],
                                                                                           config['n_ctx']))
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.n_head = config['n_head']

    def forward(self, x, mask): 
        B, T, C = x.size() # T=seq*num_item, C=emb_dim

        # batch*n_head*T*C // self.n_head
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        mask = mask.view(B, -1)
        # batch*1*1*(seq*3)
        mask = mask[:, None, None, :]
        # 1->0, 0->-10000
        mask = (1.0 - mask) * -10000.0
        # batch*n_head*T*T
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.where(self.bias[:, :, :T, :T].bool(), att, self.masked_bias.to(att.dtype))
        att = att + mask
        att = F.softmax(att, dim=-1)
        self._attn_map = att.clone()
        att = self.attn_drop(att)
        # batch*n_head*T*C // self.n_head
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

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1)
        )

        self.q2_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def save_net(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, 'QGA_critic_inverse.pt')
        torch.save(self.state_dict(), file_path)
        print(f'Model has been saved to {file_path}')

class InverseDynamicsModel(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.out_layer = nn.Linear(hidden_dim, act_dim)

    def forward(self, state, next_state):
        x = torch.cat([state, next_state], dim=-1)
        x = self.layer1(x)
        x = self.layer2(x) + x  # 残差
        x = self.layer3(x) + x  # 再次残差
        x = self.layer4(x) + x  # 再次残差
        out = self.out_layer(x)
        return out

    



class DecisionTransformer(nn.Module):

    def __init__(self, state_dim, act_dim, state_mean, state_std, hidden_size=512, action_tanh=False, K=10,
                 max_ep_len=48, scale=50,
                 target_return=1, target_ctg = 1., device="cpu",
                 baseline_method = 'vanilla_dt',
                 reweight_w = 0.2,
                 critic_ensemble = None,
                 model_ref = None,
                 learning_rate=1e-5
                 ):
        super(DecisionTransformer, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.length_times = 3
        self.reweight_w = reweight_w
        self.baseline_method = baseline_method
        self.hidden_size = 512
        self.state_mean = state_mean
        self.state_std = state_std
        self.max_length = K
        self.max_ep_len = max_ep_len

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.scale = scale
        self.target_return = target_return
        self.target_ctg = target_ctg

        self.warmup_steps = 10000
        self.weight_decay = 0.0001
        self.learning_rate = learning_rate
        self.time_dim = 8
        
        # 添加critic属性
        self.critic = None

        # ==== 新增逆动力学模型 ====
        self.id_model = InverseDynamicsModel(state_dim, act_dim, hidden_dim=256).to(self.device)
        self.id_optimizer = torch.optim.Adam(self.id_model.parameters(), lr=1e-4) 

        self.block_config = {
            "n_ctx": 1024,
            "n_embd": self.hidden_size ,  # 512
            "n_layer": 6,
            "n_head": 8,
            "n_inner": 512,
            "activation_function": "relu",
            "n_position": 1024,
            "resid_pdrop": 0.1,
            "attn_pdrop": 0.1
        }
        block_config = self.block_config
        
        self.hyperparameters = {
            "n_ctx": self.block_config['n_ctx'],
            "n_embd": self.block_config['n_embd'],
            "n_layer": self.block_config['n_layer'],
            "n_head": self.block_config['n_head'],
            "n_inner": self.block_config['n_inner'],
            "activation_function": self.block_config['activation_function'],
            "n_position": self.block_config['n_position'],
            "resid_pdrop": self.block_config['resid_pdrop'],
            "attn_pdrop": self.block_config['attn_pdrop'],
            "length_times": self.length_times,
            "hidden_size": self.hidden_size,
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "max_length": self.max_length,
            "K": K,
            "state_dim": state_dim,
            "act_dim": act_dim,
            "scale": scale,
            "target_return": target_return,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "learning_rate": self.learning_rate,
            "time_dim":self.time_dim
        }

        # n_layer of Block
        self.transformer = nn.ModuleList([Block(block_config) for _ in range(block_config['n_layer'])])

        self.embed_timestep = nn.Embedding(self.max_ep_len, self.time_dim)
        self.embed_return = torch.nn.Linear(1, self.hidden_size)
        self.embed_reward = torch.nn.Linear(1, self.hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)
        self.embed_ctg = torch.nn.Linear(1, self.hidden_size)

        self.trans_return = torch.nn.Linear(self.time_dim+self.hidden_size, self.hidden_size)
        self.trans_reward = torch.nn.Linear(self.time_dim+self.hidden_size, self.hidden_size)
        self.trans_state = torch.nn.Linear(self.time_dim+self.hidden_size, self.hidden_size)
        self.trans_action = torch.nn.Linear(self.time_dim+self.hidden_size, self.hidden_size)
        self.trans_cost = torch.nn.Linear(self.time_dim+self.hidden_size, self.hidden_size)
        self.trans_ctg = torch.nn.Linear(self.time_dim+self.hidden_size, self.hidden_size)

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

    def forward(self, states, actions, rewards, returns_to_go, ctg, score_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)

        if self.baseline_method in ['dt_reweight', 'dt_reweight_search']:
            returns_to_go  = returns_to_go + self.reweight_w * ctg
        rtg_embeddings = self.embed_return(returns_to_go)
        rewards_embeddings = self.embed_reward(rewards)
        time_embeddings = self.embed_timestep(timesteps)

        # To achieve a stable and good dt baseline, we use concat instead of common add, to let the model be aware of time
        state_embeddings = torch.cat((state_embeddings, time_embeddings), dim=-1)
        action_embeddings = torch.cat((action_embeddings, time_embeddings), dim=-1)
        rtg_embeddings = torch.cat((rtg_embeddings, time_embeddings), dim=-1)
        rewards_embeddings = torch.cat((rewards_embeddings, time_embeddings), dim=-1)

        state_embeddings = self.trans_state(state_embeddings)
        action_embeddings = self.trans_action(action_embeddings)
        rtg_embeddings = self.trans_return(rtg_embeddings)
        rewards_embeddings = self.trans_reward(rewards_embeddings)

        # batch*self.length_times*seq*dim->batch*(seq*self.length_times)*dim
        stacked_inputs = torch.stack(
            (rtg_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, self.length_times * seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # batch*(seq_len * self.length_times)*embedd_size
        stacked_attention_mask = torch.stack(
            ([attention_mask for _ in range(self.length_times)]), dim=1
        ).permute(0, 2, 1).reshape(batch_size, self.length_times * seq_length).to(stacked_inputs.dtype)

        x = stacked_inputs
        for block in self.transformer:
            x = block(x, stacked_attention_mask)

        # batch*3*seq*dim
        x = x.reshape(batch_size, seq_length, self.length_times, self.hidden_size).permute(0, 2, 1, 3)

        # predict the action based on the state embedding part
        action_preds = self.predict_action(x[:, -2])
        state_preds = self.predict_state(x[:,1])   # 预测的状态 token index为1，或者直接 x[:, 1]


        return x, action_preds, state_preds, None

    def get_action(self, states, actions, rewards, returns_to_go, ctg, score_to_go, timesteps, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        rewards = rewards.reshape(1, -1, 1)
        ctg = ctg.reshape(1, -1, 1)
        score_to_go = score_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            rewards = rewards[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]
            ctg = ctg[:, -self.max_length:]
            score_to_go = score_to_go[:, -self.max_length:]

            attention_mask = torch.cat([torch.zeros(self.max_length - states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length - states.shape[1], self.state_dim),
                             device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length - returns_to_go.shape[1], 1),
                             device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            rewards = torch.cat(
                [torch.zeros((rewards.shape[0], self.max_length - rewards.shape[1], 1), device=rewards.device),
                 rewards],
                dim=1).to(dtype=torch.float32)
            ctg = torch.cat(
                [torch.zeros((ctg.shape[0], self.max_length - ctg.shape[1], 1),
                             device=ctg.device), ctg],
                dim=1).to(dtype=torch.float32)
            score_to_go = torch.cat(
                [torch.zeros((score_to_go.shape[0], self.max_length - score_to_go.shape[1], 1),
                             device=score_to_go.device), score_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length - timesteps.shape[1]), device=timesteps.device),
                 timesteps],
                dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        x, action_preds, state_preds, idm_actions = self.forward(
            states=states, actions=actions, rewards=rewards, returns_to_go=returns_to_go, ctg=ctg, score_to_go=score_to_go, timesteps=timesteps, attention_mask=attention_mask)
        return x, action_preds[0, -1], state_preds[0, -1], idm_actions  # x is the embedding

    def take_actions(self, state, actual_excuted_action, target_return=None, target_ctg=None, pre_reward=None, pre_cost=None, cpa_constrain=None):
        self.eval()
        if self.eval_states is None:
            self.eval_states = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)
            ep_return = target_return.to(self.device) if target_return is not None else self.target_return
            self.eval_target_return = torch.tensor(ep_return, dtype=torch.float32).reshape(1, 1).to(self.device)
            self.eval_target_score_to_go = torch.tensor(ep_return, dtype=torch.float32).reshape(1,1).to(self.device)

            ep_ctg = target_ctg.to(self.device) if target_ctg is not None else self.target_ctg
            self.eval_target_ctg = torch.tensor(ep_ctg, dtype=torch.float32).reshape(1, 1).to(self.device)
        else:
            assert pre_reward is not None
            assert pre_cost is not None
            cur_state = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)
            self.eval_states = torch.cat([self.eval_states, cur_state], dim=0).to(self.device)
            
            self.eval_rewards[-1] = pre_reward
            self.eval_costs[-1] = pre_cost

            # Implementing different methods' condition to go
            pred_return = self.eval_target_return[0, -1] - (pre_reward / self.scale)
            self.eval_target_return = torch.cat([self.eval_target_return, pred_return.reshape(1, 1)], dim=1)

            # pred_ctg = self.eval_target_ctg[0, -1] - (pre_cost/ self.scale)
            pred_ctg = torch.ones_like(self.eval_target_ctg[0, -1]) # ctg is always set as 1 in the inference stage
            self.eval_target_ctg = torch.cat([self.eval_target_ctg, pred_ctg.reshape(1, 1)], dim=1)

            self.eval_timesteps = torch.cat(
                [self.eval_timesteps, torch.ones((1, 1), dtype=torch.long).to(self.device) * self.eval_timesteps[:, -1] + 1], dim=1)

        # If actual_executed_action has a value, the action actually executed should replace the placeholder action from the previous moment.
        if actual_excuted_action is None:
            self.eval_actions = torch.cat([self.eval_actions, torch.zeros(1, self.act_dim).to(self.device)], dim=0)
        else:
            self.eval_actions[-1] = torch.from_numpy(actual_excuted_action).reshape(1, self.act_dim).to(self.device)
            self.eval_actions = torch.cat([self.eval_actions, torch.zeros(1, self.act_dim).to(self.device)], dim=0)
        
        self.eval_rewards = torch.cat([self.eval_rewards, torch.zeros(1).to(self.device)])
        self.eval_costs = torch.cat([self.eval_costs, torch.zeros(1).to(self.device)])

        # states, actions, rewards, returns_to_go, ctg, score_to_go, timesteps
        x, action = self.get_action(
            (self.eval_states.to(dtype=torch.float32) - torch.tensor(self.state_mean).to(self.device)) / torch.tensor(self.state_std).to(self.device),
            self.eval_actions.to(dtype=torch.float32),
            self.eval_rewards.to(dtype=torch.float32),
            self.eval_target_return.to(dtype=torch.float32),
            self.eval_target_ctg.to(dtype=torch.float32),
            self.eval_target_score_to_go.to(dtype=torch.float32),
            self.eval_timesteps.to(dtype=torch.long),
        )
        self.eval_actions[-1] = action
        action = action.detach().cpu().numpy()
        return action

    def take_actions_mrtg(self, state, actual_excuted_action, target_return=None, target_ctg=None, pre_reward=None, pre_cost=None, cpa_constrain=None, m=5):
        self.eval()
        # 初始化状态
        if self.eval_states is None:
            self.eval_states = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)
            ep_return = target_return.to(self.device) if target_return is not None else self.target_return
            self.eval_target_return = torch.tensor(ep_return, dtype=torch.float32).reshape(1, 1).to(self.device)
            self.eval_target_score_to_go = torch.tensor(ep_return, dtype=torch.float32).reshape(1,1).to(self.device)

            ep_ctg = target_ctg.to(self.device) if target_ctg is not None else self.target_ctg
            self.eval_target_ctg = torch.tensor(ep_ctg, dtype=torch.float32).reshape(1, 1).to(self.device)
            
            # 如果是初始状态，直接返回动作
            x, action = self.get_action(
                (self.eval_states.to(dtype=torch.float32) - torch.tensor(self.state_mean).to(self.device)) / torch.tensor(self.state_std).to(self.device),
                torch.zeros((1, self.act_dim), dtype=torch.float32).to(self.device),
                torch.zeros(1, dtype=torch.float32).to(self.device),
                self.eval_target_return.to(dtype=torch.float32),
                self.eval_target_ctg.to(dtype=torch.float32),
                self.eval_target_score_to_go.to(dtype=torch.float32),
                torch.tensor(0, dtype=torch.long).reshape(1, 1).to(self.device),
            )
            self.eval_actions = torch.zeros((1, self.act_dim), dtype=torch.float32).to(self.device)
            self.eval_actions[0] = action
            self.eval_rewards = torch.zeros(1, dtype=torch.float32).to(self.device)
            self.eval_costs = torch.zeros(1, dtype=torch.float32).to(self.device)
            self.eval_timesteps = torch.tensor(0, dtype=torch.long).reshape(1, 1).to(self.device)
            
            action = action.detach().cpu().numpy()
            return action
        else:
            assert pre_reward is not None
            assert pre_cost is not None
            
            # 更新历史数据
            if actual_excuted_action is not None:
                self.eval_actions[-1] = torch.from_numpy(actual_excuted_action).reshape(1, self.act_dim).to(self.device)
            
            self.eval_rewards[-1] = pre_reward
            self.eval_costs[-1] = pre_cost
            
            # 当前状态
            cur_state = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)
            
            # 实现多RTG候选动作采样和Q值计算
            candidate_actions = []
            q_values = []
            
            # 采样多个不同的return-to-go值
            rtg_candidates = []
            base_rtg = self.eval_target_return[0, -1] - (pre_reward / self.scale)
            rtg_range = 0.3 * base_rtg  # RTG范围
            for i in range(m):
                # 在base_rtg周围生成不同的RTG值
                rtg_i = base_rtg + (2 * rtg_range * i / (m-1 if m > 1 else 1))
                rtg_candidates.append(rtg_i)
            
            # 为每个候选RTG生成动作
            for i in range(m):
                # 复制当前状态和历史
                temp_states = self.eval_states.clone()
                temp_states = torch.cat([temp_states, cur_state], dim=0)
                
                temp_actions = self.eval_actions.clone()
                temp_actions = torch.cat([temp_actions, torch.zeros(1, self.act_dim).to(self.device)], dim=0)
                
                temp_rewards = self.eval_rewards.clone()
                temp_rewards = torch.cat([temp_rewards, torch.zeros(1).to(self.device)])
                
                temp_costs = self.eval_costs.clone()
                temp_costs = torch.cat([temp_costs, torch.zeros(1).to(self.device)])
                
                temp_timesteps = self.eval_timesteps.clone()
                temp_timesteps = torch.cat([temp_timesteps, torch.ones((1, 1), dtype=torch.long).to(self.device) * temp_timesteps[:, -1] + 1], dim=1)
                
                # 使用候选RTG
                temp_target_return = self.eval_target_return.clone()
                temp_target_return = torch.cat([temp_target_return, rtg_candidates[i].reshape(1, 1)], dim=1)
                
                # CTG保持不变
                temp_target_ctg = self.eval_target_ctg.clone()
                pred_ctg = torch.ones_like(self.eval_target_ctg[0, -1])
                temp_target_ctg = torch.cat([temp_target_ctg, pred_ctg.reshape(1, 1)], dim=1)
                
                # 使用当前模型生成动作
                x, action = self.get_action(
                    (temp_states.to(dtype=torch.float32) - torch.tensor(self.state_mean).to(self.device)) / torch.tensor(self.state_std).to(self.device),
                    temp_actions.to(dtype=torch.float32),
                    temp_rewards.to(dtype=torch.float32),
                    temp_target_return.to(dtype=torch.float32),
                    temp_target_ctg.to(dtype=torch.float32),
                    temp_target_return.to(dtype=torch.float32),  # 使用RTG作为score_to_go
                    temp_timesteps.to(dtype=torch.long),
                )
                
                # 使用critic计算Q值（如果存在）
                if self.critic is not None:
                    # 标准化状态
                    normalized_state = (cur_state.to(dtype=torch.float32) - torch.tensor(self.state_mean).to(self.device,dtype=torch.float32)) / torch.tensor(self.state_std).to(self.device,dtype=torch.float32)
                    # 确保action和normalized_state具有相同的数据类型
                    action_for_critic = action.unsqueeze(0).to(dtype=torch.float32)
                    
                    # 使用critic计算Q值
                    q1, q2 = self.critic(normalized_state, action_for_critic)
                    # 使用两个Q网络中较小的值作为Q值估计
                    q_value = torch.min(q1, q2)
                else:
                    exit(0)
                
                candidate_actions.append(action)
                q_values.append(q_value.detach())
            
            # 选择Q值最大的动作
            q_values = torch.stack(q_values).to(self.device)
            max_q_idx = torch.argmax(q_values).item()
            best_action = candidate_actions[max_q_idx]
            best_rtg = rtg_candidates[max_q_idx]
            
            # 更新状态历史
            self.eval_states = torch.cat([self.eval_states, cur_state], dim=0)
            self.eval_actions = torch.cat([self.eval_actions, torch.zeros(1, self.act_dim).to(self.device)], dim=0)
            self.eval_actions[-1] = best_action
            self.eval_rewards = torch.cat([self.eval_rewards, torch.zeros(1).to(self.device)])
            self.eval_costs = torch.cat([self.eval_costs, torch.zeros(1).to(self.device)])
            
            # 更新RTG
            self.eval_target_return = torch.cat([self.eval_target_return, best_rtg.reshape(1, 1)], dim=1)
            
            # 更新CTG
            pred_ctg = torch.ones_like(self.eval_target_ctg[0, -1])
            self.eval_target_ctg = torch.cat([self.eval_target_ctg, pred_ctg.reshape(1, 1)], dim=1)
            
            # 更新时间步
            self.eval_timesteps = torch.cat(
                [self.eval_timesteps, torch.ones((1, 1), dtype=torch.long).to(self.device) * self.eval_timesteps[:, -1] + 1], dim=1)
            
            return best_action.detach().cpu().numpy()

    def take_actions_mrtg_naction(self, state, actual_excuted_action, target_return=None, target_ctg=None, pre_reward=None, pre_cost=None, cpa_constrain=None, m=5, n=5, noise_std=0.1):
        """
        Args:
            m: 多个RTG备选数量，每个rtg都采样一次
            n: 每个rtg下对主动作加高斯噪声采样数量
            noise_std: 高斯噪声的标准差
        """
        self.eval()
        # 初始化状态
        if self.eval_states is None:
            self.eval_states = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)
            ep_return = target_return.to(self.device) if target_return is not None else self.target_return
            self.eval_target_return = torch.tensor(ep_return, dtype=torch.float32).reshape(1, 1).to(self.device)
            self.eval_target_score_to_go = torch.tensor(ep_return, dtype=torch.float32).reshape(1,1).to(self.device)

            ep_ctg = target_ctg.to(self.device) if target_ctg is not None else self.target_ctg
            self.eval_target_ctg = torch.tensor(ep_ctg, dtype=torch.float32).reshape(1, 1).to(self.device)
            x, action = self.get_action(
                (self.eval_states.to(dtype=torch.float32) - torch.tensor(self.state_mean).to(self.device)) / torch.tensor(self.state_std).to(self.device),
                torch.zeros((1, self.act_dim), dtype=torch.float32).to(self.device),
                torch.zeros(1, dtype=torch.float32).to(self.device),
                self.eval_target_return.to(dtype=torch.float32),
                self.eval_target_ctg.to(dtype=torch.float32),
                self.eval_target_score_to_go.to(dtype=torch.float32),
                torch.tensor(0, dtype=torch.long).reshape(1, 1).to(self.device),
            )
            self.eval_actions = torch.zeros((1, self.act_dim), dtype=torch.float32).to(self.device)
            self.eval_actions[0] = action
            self.eval_rewards = torch.zeros(1, dtype=torch.float32).to(self.device)
            self.eval_costs = torch.zeros(1, dtype=torch.float32).to(self.device)
            self.eval_timesteps = torch.tensor(0, dtype=torch.long).reshape(1, 1).to(self.device)

            action = action.detach().cpu().numpy()
            return action
        else:
            assert pre_reward is not None
            assert pre_cost is not None

            # 更新历史数据
            if actual_excuted_action is not None:
                self.eval_actions[-1] = torch.from_numpy(actual_excuted_action).reshape(1, self.act_dim).to(self.device)
            
            self.eval_rewards[-1] = pre_reward
            self.eval_costs[-1] = pre_cost

            cur_state = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)

            candidate_actions = []
            q_values = []
            chosen_rtgs     = []

            # 采样多个不同的return-to-go值
            rtg_candidates = []
            base_rtg = self.eval_target_return[0, -1] - (pre_reward / self.scale)
            rtg_range = 0.2 * base_rtg  # RTG范围
            for i in range(m):
                rtg_i = base_rtg + (2 * rtg_range * i / (m-1 if m > 1 else 1))
                rtg_candidates.append(rtg_i)

            # ====== 新增：每个rtg可采样多个噪声动作 ======
            for i in range(m):
                temp_states = self.eval_states.clone()
                temp_states = torch.cat([temp_states, cur_state], dim=0)

                temp_actions = self.eval_actions.clone()
                temp_actions = torch.cat([temp_actions, torch.zeros(1, self.act_dim).to(self.device)], dim=0)

                temp_rewards = self.eval_rewards.clone()
                temp_rewards = torch.cat([temp_rewards, torch.zeros(1).to(self.device)])

                temp_costs = self.eval_costs.clone()
                temp_costs = torch.cat([temp_costs, torch.zeros(1).to(self.device)])

                temp_timesteps = self.eval_timesteps.clone()
                temp_timesteps = torch.cat([temp_timesteps, torch.ones((1, 1), dtype=torch.long).to(self.device) * temp_timesteps[:, -1] + 1], dim=1)

                temp_target_return = self.eval_target_return.clone()
                temp_target_return = torch.cat([temp_target_return, rtg_candidates[i].reshape(1, 1)], dim=1)

                temp_target_ctg = self.eval_target_ctg.clone()
                pred_ctg = torch.ones_like(self.eval_target_ctg[0, -1])
                temp_target_ctg = torch.cat([temp_target_ctg, pred_ctg.reshape(1, 1)], dim=1)

                # 先取主动作
                x, action = self.get_action(
                    (temp_states.to(dtype=torch.float32) - torch.tensor(self.state_mean).to(self.device)) / torch.tensor(self.state_std).to(self.device),
                    temp_actions.to(dtype=torch.float32),
                    temp_rewards.to(dtype=torch.float32),
                    temp_target_return.to(dtype=torch.float32),
                    temp_target_ctg.to(dtype=torch.float32),
                    temp_target_return.to(dtype=torch.float32),  # 用RTG做score_to_go
                    temp_timesteps.to(dtype=torch.long),
                )

                # ====== 高斯扰动动作采样（含原动作） ======
                for j in range(n):
                    if j == 0:
                        # 加入原始动作
                        action_perturbed = action.clone()
                    else:
                        # 高斯噪声叠加
                        noise = torch.randn_like(action) * noise_std
                        action_perturbed = action + noise

                    if self.critic is not None:
                        normalized_state = (cur_state.to(dtype=torch.float32) - torch.tensor(self.state_mean).to(self.device,dtype=torch.float32)) / torch.tensor(self.state_std).to(self.device,dtype=torch.float32)
                        action_for_critic = action_perturbed.unsqueeze(0).to(dtype=torch.float32)
                        q1, q2 = self.critic(normalized_state, action_for_critic)
                        q_value = torch.min(q1, q2)
                    else:
                        exit(0)

                    candidate_actions.append(action_perturbed)
                    q_values.append(q_value.detach())
                    chosen_rtgs.append(rtg_candidates[i])

            # 选择Q值最大的动作
            q_values = torch.stack(q_values).to(self.device)
            max_q_idx = torch.argmax(q_values).item()
            best_action = candidate_actions[max_q_idx]
            best_rtg = chosen_rtgs[max_q_idx]

            # 更新状态历史（与原逻辑基本一致，只best_action/best_rtg是多重采样里选出来的）
            self.eval_states = torch.cat([self.eval_states, cur_state], dim=0)
            self.eval_actions = torch.cat([self.eval_actions, torch.zeros(1, self.act_dim).to(self.device)], dim=0)
            self.eval_actions[-1] = best_action
            self.eval_rewards = torch.cat([self.eval_rewards, torch.zeros(1).to(self.device)])
            self.eval_costs = torch.cat([self.eval_costs, torch.zeros(1).to(self.device)])

            self.eval_target_return = torch.cat([self.eval_target_return, best_rtg.reshape(1, 1)], dim=1)
            pred_ctg = torch.ones_like(self.eval_target_ctg[0, -1])
            self.eval_target_ctg = torch.cat([self.eval_target_ctg, pred_ctg.reshape(1, 1)], dim=1)

            self.eval_timesteps = torch.cat(
                [self.eval_timesteps, torch.ones((1, 1), dtype=torch.long).to(self.device) * self.eval_timesteps[:, -1] + 1], dim=1)

            return best_action.detach().cpu().numpy()

    def take_action_inverse(self, state, actual_executed_action=None, target_return=None, target_ctg=None, pre_reward=None, pre_cost=None, cpa_constrain=None):
        """
        用Decision Transformer和IDM分别产生action，利用Critic选Q高的那个。
        """
        self.eval()
        device = self.device

        # -------- 初始化历史记录 -------- #
        if self.eval_states is None:
            # 列表记录方式，初始化历史
            self.eval_states = torch.from_numpy(state).reshape(1, self.state_dim).to(device)
            ep_return = target_return.to(device) if target_return is not None else self.target_return
            self.eval_target_return = torch.tensor(ep_return, dtype=torch.float32).reshape(1, 1).to(device)
            self.eval_target_score_to_go = torch.tensor(ep_return, dtype=torch.float32).reshape(1,1).to(device)

            ep_ctg = target_ctg.to(device) if target_ctg is not None else self.target_ctg
            self.eval_target_ctg = torch.tensor(ep_ctg, dtype=torch.float32).reshape(1, 1).to(device)

            self.eval_actions = torch.zeros((1, self.act_dim), dtype=torch.float32).to(device)
            self.eval_rewards = torch.zeros(1, dtype=torch.float32).to(device)
            self.eval_costs = torch.zeros(1, dtype=torch.float32).to(device)
            self.eval_timesteps = torch.tensor(0, dtype=torch.long).reshape(1, 1).to(device)

        else:
            # 已有历史
            assert pre_reward is not None
            assert pre_cost is not None
            cur_state = torch.from_numpy(state).reshape(1, self.state_dim).to(device)
            self.eval_states = torch.cat([self.eval_states, cur_state], dim=0).to(device)

            if actual_executed_action is not None:
                self.eval_actions[-1] = torch.from_numpy(actual_executed_action).reshape(1, self.act_dim).to(device)
            self.eval_actions = torch.cat([self.eval_actions, torch.zeros(1, self.act_dim).to(device)], dim=0)
            self.eval_rewards[-1] = pre_reward
            self.eval_rewards = torch.cat([self.eval_rewards, torch.zeros(1).to(device)])
            self.eval_costs[-1] = pre_cost
            self.eval_costs = torch.cat([self.eval_costs, torch.zeros(1).to(device)])

            # 更新RTG和CTG
            pred_return = self.eval_target_return[0, -1] - (pre_reward / self.scale)
            self.eval_target_return = torch.cat([self.eval_target_return, pred_return.reshape(1, 1)], dim=1)
            pred_ctg = torch.ones_like(self.eval_target_ctg[0, -1])
            self.eval_target_ctg = torch.cat([self.eval_target_ctg, pred_ctg.reshape(1, 1)], dim=1)

            self.eval_timesteps = torch.cat(
                [self.eval_timesteps, torch.ones((1, 1), dtype=torch.long).to(device) * self.eval_timesteps[:, -1] + 1], dim=1)

            self.eval_target_score_to_go = torch.cat([self.eval_target_score_to_go, pred_return.reshape(1,1)], dim=1)

        # ----------- (1) Decision Transformer 动作 ----------- #
        normalized_states = (self.eval_states.to(dtype=torch.float32) - torch.tensor(self.state_mean).to(device)) / torch.tensor(self.state_std).to(device)
        actions_seq = self.eval_actions.to(dtype=torch.float32)
        rewards_seq = self.eval_rewards.to(dtype=torch.float32)
        target_returns = self.eval_target_return.to(dtype=torch.float32)
        target_ctg = self.eval_target_ctg.to(dtype=torch.float32)
        score_to_go = self.eval_target_score_to_go.to(dtype=torch.float32)
        timesteps_seq = self.eval_timesteps.to(dtype=torch.long)

        _, action_dt, _, _ = self.get_action(normalized_states, actions_seq, rewards_seq, target_returns, target_ctg, score_to_go, timesteps_seq)
        action_dt = action_dt.unsqueeze(0)    # (1, act_dim)

        # ----------- (2) IDM产生的action ----------- #
        # 需要(state_{t-1}, state_t)做输入（即历史倒数第二和最后一个状态），target为action_{t}
        if self.eval_states.shape[0] >= 2:
            state_prev = normalized_states[-2].unsqueeze(0)  # shape: (1, state_dim)
            state_cur = normalized_states[-1].unsqueeze(0)
            state_prev = state_prev.to(dtype=torch.float32)
            state_cur  = state_cur.to(dtype=torch.float32)
            action_idm = self.id_model(state_prev, state_cur)   # (1, act_dim)
        else:
            # 没有倒数第二状态，IDM没法工作，用DT动作或者随机动作代替
            action_idm = action_dt.clone()

        # ----------- (3) 用Critic判断哪个动作Q值高 ----------- #
        # 最新状态（归一化后的）
        last_state = normalized_states[-1].unsqueeze(0).to(dtype=torch.float32)   # (1, state_dim)

        
        action_dt = action_dt.to(dtype=torch.float32)
        action_idm = action_idm.to(dtype=torch.float32)

        # 用critic算Q值
        with torch.no_grad():
            if self.critic is not None:
                q1_dt, q2_dt = self.critic(last_state, action_dt)
                q_dt = torch.min(q1_dt, q2_dt)
                q1_idm, q2_idm = self.critic(last_state, action_idm)
                q_idm = torch.min(q1_idm, q2_idm)
            else:
                # 没critic，只能用DT
                return action_dt.squeeze(0).detach().cpu().numpy()
        
        # Debug: 打印两个动作的值和Q值
        # print(f"[DEBUG] DT action: {action_dt.squeeze(0).detach().cpu().numpy()}, Q(DT)={q_dt.item():.6f}")
        # print(f"[DEBUG] IDM action: {action_idm.squeeze(0).detach().cpu().numpy()}, Q(IDM)={q_idm.item():.6f}")

        # # 对比并选最终动作
        # if q_dt.item() >= q_idm.item():
        #     chosen_action = action_dt
        #     chosen_name = "DT"
        # else:
        #     chosen_action = action_idm
        #     chosen_name = "IDM"

        # # 只用DT
        # chosen_action = action_dt
        # chosen_name = "DT"

        # # 只用IDM
        # chosen_action = action_idm
        # chosen_name = "IDM"

        # 随机选取
        import random
        choices = [
            (action_dt, "DT"),
            (action_idm, "IDM")
        ]
        chosen_action, chosen_name = random.choice(choices)


        # print(f"[DEBUG] Chosen: {chosen_name}; value: {chosen_action.squeeze(0).detach().cpu().numpy()}")


        # ----------- (4) 填写到动作历史，并返回 ----------- #
        self.eval_actions[-1] = chosen_action
        return chosen_action.squeeze(0).detach().cpu().numpy()


    def init_eval(self):
        self.eval_states = None
        self.eval_actions = torch.zeros((0, self.act_dim), dtype=torch.float32).to(self.device)
        self.eval_rewards = torch.zeros(0, dtype=torch.float32).to(self.device)
        self.eval_costs = torch.zeros(0, dtype=torch.float32).to(self.device)

        self.eval_target_return = None
        self.eval_target_ctg = None

        self.eval_timesteps = torch.tensor(0, dtype=torch.long).reshape(1, 1).to(self.device)

        self.eval_episode_return, self.eval_episode_length = 0, 0

    def save_net(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, 'QGA_dt_inverse.pt')
        torch.save(self.state_dict(), file_path)
        print(f'Model has been saved to {file_path}')

    def save_jit(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        jit_model = torch.jit.script(self.cpu())
        torch.jit.save(jit_model, f'{save_path}/dt_model.pth')

    def load_net(self, load_path="saved_model/DT/dt.pt", device='cuda', critic_path=None, idm_path=None):
        file_path = load_path
        self.load_state_dict(torch.load(file_path, map_location=device), strict=False)
        print(f'Model loaded from {load_path} to device {self.device}')
        
        # 如果提供了critic路径，则加载critic模型
        if critic_path is not None:
            # 初始化critic模型
            self.critic = Critic(self.state_dim, self.act_dim).to(self.device)
            # 加载critic模型参数
            self.critic.load_state_dict(torch.load(critic_path, map_location=device))
            # 设置为评估模式
            self.critic.eval()
            print(f'Critic model loaded from {critic_path} to device {self.device}')
            print(f'Critic model device: {next(self.critic.parameters()).device}')

        # 如果提供了IDM路径，则加载IDM模型参数
        if idm_path is not None:
            # 初始化idm模型（如果尚未初始化）            
            if not hasattr(self, "id_model") or self.id_model is None:
                self.id_model = InverseDynamicsModel(self.state_dim, self.act_dim, hidden_dim=256).to(self.device)
            # 加载参数
            self.id_model.load_state_dict(torch.load(idm_path, map_location=device))
            self.id_model.eval()
            print(f'IDM model loaded from {idm_path} to device {self.device}')
            print(f'IDM model device: {next(self.id_model.parameters()).device}')
    
    def save_idm(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, 'QGA_idm.pt')
        torch.save(self.id_model.state_dict(), file_path)
        print(f'ID Model has been saved to {file_path}')
