import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math


class Q(nn.Module):
    '''
        IQL-Q net
    '''

    def __init__(self, dim_observation, dim_action, hidden_size=512):
        super(Q, self).__init__()
        self.dim_observation = hidden_size
        self.dim_action = hidden_size

        self.obs_FC = nn.Linear(self.dim_observation, 64)
        self.action_FC = nn.Linear( self.dim_action , 64)
        self.FC1 = nn.Linear(128, hidden_size)
        self.FC2 = nn.Linear(hidden_size, hidden_size)
        self.FC3 = nn.Linear(hidden_size, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        obs_embedding = self.obs_FC(obs)
        action_embedding = self.action_FC(acts)
        embedding = torch.cat([obs_embedding, action_embedding], dim=-1)
        q = self.FC3(F.relu(self.FC2(F.relu(self.FC1(embedding)))))
        return q


class V(nn.Module):
    '''
        IQL-V net
    '''
    def __init__(self, dim_observation, hidden_size=512):
        super(V, self).__init__()
        self.FC1 = nn.Linear(hidden_size, 128)
        self.FC2 = nn.Linear(128, hidden_size)
        self.FC3 = nn.Linear(hidden_size, hidden_size)
        self.FC4 = nn.Linear(hidden_size, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.relu(self.FC3(result))
        return self.FC4(result)

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

    def forward(self, x, mask):  # batch*(seq*3)*dim, batch*(seq*3)
        B, T, C = x.size()  # T=seq*3, C=dim

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

    def forward(self, inputs_embeds, attention_mask):  # batch*(seq*3)*dim, batch*(seq*3)
        x = inputs_embeds + self.attn(self.ln1(inputs_embeds), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class DT_Critic(nn.Module):

    def __init__(self, state_dim, act_dim, state_mean, state_std, hidden_size=512, action_tanh=False, K=48,
                 max_ep_len=48, scale=50,
                 target_return=1., target_ctg=1., device="cpu",
                 baseline_method='vanilla_dt',
                 reweight_w=0.2,
                 use_rtg=True,
                 ):
        super(DT_Critic, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.use_rtg = use_rtg

        self.length_times = 3
        self.baseline_method = baseline_method
        self.reweight_w = reweight_w

        self.hidden_size = 512
        self.state_mean = state_mean
        self.state_std = state_std
        # assert self.hidden_size == config['n_embd']
        self.max_length = K
        self.max_ep_len = max_ep_len

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.scale = scale
        self.target_return = target_return
        self.target_ctg = target_ctg

        self.warmup_steps = 10000
        self.weight_decay = 0.0001
        self.learning_rate = 1e-4
        self.time_dim = self.hidden_size

        self.block_config = {
            "n_ctx": 1024,
            "n_embd": self.hidden_size,  # 512
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
            # "state_max": self.state_max,
            # "state_min": self.state_min,
            "max_length": self.max_length,
            "K": K,
            "state_dim": state_dim,
            "act_dim": act_dim,
            "scale": scale,
            "target_return": target_return,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "learning_rate": self.learning_rate,
            "time_dim": self.time_dim

        }

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>> Transformer backbone parameters >>>>>>>>>>>>>>>>>>>>>>>>>>
        # n_layer of Block
        self.transformer = nn.ModuleList([Block(block_config) for _ in range(block_config['n_layer'])])

        self.embed_timestep = nn.Embedding(self.max_ep_len, self.time_dim)
        self.embed_return = torch.nn.Linear(1, self.hidden_size)
        self.embed_reward = torch.nn.Linear(1, self.hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, self.hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, self.hidden_size)
        self.embed_ctg = torch.nn.Linear(1, self.hidden_size)
        # self.embed_costs = torch.nn.Linear(1, self.hidden_size)

        self.trans_return = torch.nn.Linear(self.time_dim + self.hidden_size, self.hidden_size)
        self.trans_reward = torch.nn.Linear(self.time_dim + self.hidden_size, self.hidden_size)
        self.trans_state = torch.nn.Linear(self.time_dim + self.hidden_size, self.hidden_size)
        self.trans_action = torch.nn.Linear(self.time_dim + self.hidden_size, self.hidden_size)
        self.trans_cost = torch.nn.Linear(self.time_dim + self.hidden_size, self.hidden_size)
        self.trans_ctg = torch.nn.Linear(self.time_dim + self.hidden_size, self.hidden_size)

        self.embed_ln = nn.LayerNorm(self.hidden_size)

        # add IQL network
        self.expectile = 0.7
        self.value_net = V(self.hidden_size)
        self.critic1 = Q(self.hidden_size, self.act_dim)
        self.critic2 = Q(self.hidden_size, self.act_dim)
        self.critic1_target = Q(self.hidden_size, self.act_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = Q(self.hidden_size, self.act_dim)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.GAMMA = 0.99
        self.tau = 0.01

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                            lambda steps: min((steps + 1) / self.warmup_steps, 1))
        self.init_eval()

    def l2_loss(self, diff, expectile=0.8):
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def calc_value_loss(self, states, actions, attention_mask=None):
        with torch.no_grad():
            q1 = self.critic1_target(states, actions)
            q2 = self.critic2_target(states, actions)

            q1 = q1.reshape(-1, q1.shape[-1])[attention_mask.reshape(-1) > 0]
            q2 = q2.reshape(-1, q2.shape[-1])[attention_mask.reshape(-1) > 0]

            min_Q = torch.min(q1, q2)

        value = self.value_net(states)
        value = value.reshape(-1, q1.shape[-1])[attention_mask.reshape(-1) > 0]

        value_loss = self.l2_loss(min_Q - value, self.expectile).mean()
        return value_loss

    def calc_q_loss(self, states, actions, rewards, dones, next_states, attention_mask=None):
        with torch.no_grad():
            next_v = self.value_net(next_states)
            q_target = rewards + (self.GAMMA * (1 - dones.unsqueeze(-1)) * next_v)

        q1 = self.critic1(states, actions)
        q2 = self.critic2(states, actions)

        q1  = q1.reshape(-1, q1.shape[-1])[attention_mask.reshape(-1) > 0]
        q2 = q2.reshape(-1, q2.shape[-1])[attention_mask.reshape(-1) > 0]
        q_target = q_target.reshape(-1, q2.shape[-1])[attention_mask.reshape(-1) > 0]

        critic1_loss = ((q1 - q_target) ** 2).mean()
        critic2_loss = ((q2 - q_target) ** 2).mean()
        return critic1_loss, critic2_loss

    def update_target(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_((1. - self.tau) * target_param.data + self.tau * local_param.data)

    def forward(self, states, actions, rewards, returns_to_go, ctg, score_to_go, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        # If rtg is not needed for training, set returns_to_go to zero, then obtain rtg_embeddings and input them into DT_critic. >>>>>>>>>>
        if not self.use_rtg:
            returns_to_go = torch.zeros_like(returns_to_go)
            rtg_embeddings = self.embed_return(
                returns_to_go)  # just a placeholder to keep the critic model the same structure as the policy DT model, because the returns_to_go is zeros here
        else:
            returns_to_go = returns_to_go + self.reweight_w * ctg
            #     # if 'only_ctg' in self.baseline_method:
            #     #     returns_to_go =  ctg
            rtg_embeddings = self.embed_return(returns_to_go)

        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = torch.cat((state_embeddings, time_embeddings), dim=-1)
        action_embeddings = torch.cat((action_embeddings, time_embeddings), dim=-1)
        rtg_embeddings = torch.cat((rtg_embeddings, time_embeddings), dim=-1)

        state_embeddings = self.trans_state(state_embeddings)
        action_embeddings = self.trans_action(action_embeddings)
        rtg_embeddings = self.trans_return(rtg_embeddings)

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
        # state_action_embedding = torch.cat((x[:, 1], x[:, 2]), dim=-1)
        return_embedding = x[:, 0]
        state_embedding = x[:,1]
        action_embedding = x[:, 2]

        q_target_1 = self.critic1_target(state_embedding, action_embedding)
        q_target_2 = self.critic2_target(state_embedding, action_embedding)
        return_value = [q_target_1, q_target_2]
        

        return return_embedding, state_embedding, action_embedding, return_value

    def get_critic(self, states, actions, rewards, returns_to_go, ctg, score_to_go, timesteps, **kwargs):
        # we don't care about the past rewards in this model
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

        return_embedding, state_embedding, action_embedding, return_value = self.forward(
            states=states, actions=actions, rewards=rewards, returns_to_go=returns_to_go, ctg=ctg,
            score_to_go=score_to_go, timesteps=timesteps, attention_mask=attention_mask)
        return return_value

    def step(self, states, actions, rewards, dones, rtg, timesteps, attention_mask, ctg, score_to_go, costs):
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        costs = costs.to(self.device)  # cost is cost(s_t, a_t), which is every single step's true cost
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        timesteps = timesteps.to(self.device)
        attention_mask = attention_mask.to(self.device)
        ctg = ctg.to(self.device)
        score_to_go = score_to_go.to(self.device)

        self.optimizer.zero_grad()
        return_embedding, state_embedding, action_embedding, return_value = self.forward(
        states=states, actions=actions, rewards=rewards, returns_to_go=rtg[:, :-1], ctg=ctg[:, :-1],
        score_to_go=score_to_go[:, :-1], timesteps=timesteps, attention_mask=attention_mask,
        )

        value_loss = self.calc_value_loss(state_embedding, action_embedding, attention_mask)

        states_dt_critic = state_embedding[:,:-1]
        next_states_dt_critic = state_embedding[:, 1:]
        actions_critic = action_embedding[:, :-1]
        rewards_critic = rewards[:, :-1]
        costs_critic = costs[:, :-1]
        cpa_real = costs_critic / (rewards_critic + 1e-10)
        cpa_term = torch.clip((1 / (cpa_real+1e-5)), min = 0., max = 1.)

        rewards_critic = rewards_critic + self.reweight_w * cpa_term

        dones_critic = dones[:, :-1]
        attention_mask_critic = attention_mask[:,  :-1]
        critic1_loss, critic2_loss = self.calc_q_loss(states_dt_critic, actions_critic, rewards_critic, dones_critic, next_states_dt_critic, attention_mask_critic)

        loss = value_loss + critic1_loss + critic2_loss
        loss.backward()

        self.optimizer.step()

        self.update_target(self.critic1, self.critic1_target)
        self.update_target(self.critic2, self.critic2_target)

        return critic1_loss.detach().cpu().item(), critic2_loss.detach().cpu().item(), value_loss.detach().cpu().item()

    def take_critics(self, state, action, whether_concat_new, target_return=None, target_ctg=None, pre_reward=None,
                     pre_cost=None, cpa_constrain=None):
        self.eval()
        if self.eval_states is None:
            self.eval_states = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)
            self.eval_actions = torch.from_numpy(action).reshape(1, self.act_dim).to(self.device)

            ep_return = target_return.to(self.device) if target_return is not None else self.target_return
            self.eval_target_return = torch.tensor(ep_return, dtype=torch.float32).reshape(1, 1).to(self.device)
            self.eval_target_score_to_go = torch.tensor(ep_return, dtype=torch.float32).reshape(1, 1).to(self.device)

            ep_ctg = target_ctg.to(self.device) if target_ctg is not None else self.target_ctg
            self.eval_target_ctg = torch.tensor(ep_ctg, dtype=torch.float32).reshape(1, 1).to(self.device)
        else:
            if whether_concat_new:
                assert pre_reward is not None
                assert pre_cost is not None
                cur_state = torch.from_numpy(state).reshape(1, self.state_dim).to(self.device)
                self.eval_states = torch.cat([self.eval_states, cur_state], dim=0).to(self.device)

                cur_action = torch.from_numpy(action).reshape(1, self.act_dim).to(self.device)
                self.eval_actions = torch.cat([self.eval_actions, cur_action], dim=0).to(self.device)

                self.eval_rewards[-1] = pre_reward
                self.eval_costs[-1] = pre_cost

                pred_return = self.eval_target_return[0, -1] - (pre_reward / self.scale)
                self.eval_target_return = torch.cat([self.eval_target_return, pred_return.reshape(1, 1)], dim=1)

                # pred_ctg = self.eval_target_ctg[0, -1] - (pre_cost/ self.scale)
                pred_ctg = torch.ones_like(self.eval_target_ctg[0, -1])  # ctg is always set as 1 in the inference stage
                self.eval_target_ctg = torch.cat([self.eval_target_ctg, pred_ctg.reshape(1, 1)], dim=1)

                self.eval_timesteps = torch.cat(
                    [self.eval_timesteps,
                     torch.ones((1, 1), dtype=torch.long).to(self.device) * self.eval_timesteps[:, -1] + 1], dim=1)
            else:
                cur_action = torch.from_numpy(action).reshape(1, self.act_dim).to(self.device)
                self.eval_actions[-1] = cur_action

        if whether_concat_new:
            self.eval_rewards = torch.cat([self.eval_rewards, torch.zeros(1).to(self.device)])
            self.eval_costs = torch.cat([self.eval_costs, torch.zeros(1).to(self.device)])

        assess_values = self.get_critic(
            (self.eval_states.to(dtype=torch.float32) - torch.tensor(self.state_mean).to(self.device)) / torch.tensor(
                self.state_std).to(self.device),
            self.eval_actions.to(dtype=torch.float32),
            self.eval_rewards.to(dtype=torch.float32),
            self.eval_target_return.to(dtype=torch.float32),
            self.eval_target_ctg.to(dtype=torch.float32),
            self.eval_target_score_to_go.to(dtype=torch.float32),
            self.eval_timesteps.to(dtype=torch.long),
        )

        return assess_values  

    def init_eval(self):
        self.eval_states = None
        self.eval_actions = None
        self.eval_rewards = torch.zeros(0, dtype=torch.float32).to(self.device)
        self.eval_costs = torch.zeros(0, dtype=torch.float32).to(self.device)

        self.eval_target_return = None
        self.eval_target_ctg = None

        self.eval_timesteps = torch.tensor(0, dtype=torch.long).reshape(1, 1).to(self.device)

        self.eval_episode_return, self.eval_episode_length = 0, 0

    def save_net(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, 'dt_reweight_w_0_2_scale_50.pt')
        torch.save(self.state_dict(), file_path)

    def save_jit(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        jit_model = torch.jit.script(self.cpu())
        torch.jit.save(jit_model, f'{save_path}/dt_model.pth')

    def load_net(self, load_path="saved_model/DT/dt.pt", device='cpu'):
        file_path = load_path
        self.load_state_dict(torch.load(file_path, map_location=device))