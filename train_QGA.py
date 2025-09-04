import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import numpy as np
from bidding_train_env.common.utils import save_normalize_dict
from bidding_train_env.baseline.gas.utils import EpisodeReplayBuffer
from bidding_train_env.baseline.gas_dt_Q_regularized.dt_baselines import DecisionTransformer, Critic
from torch.utils.data import DataLoader, WeightedRandomSampler
import logging
import os
import torch
import argparse
import torch.nn.functional as F

os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")

def run_dt_baselines(baseline_method='dt_reweight', reweight_w=0.2, data_path=None, sparse_data=False):
    writer = SummaryWriter(f"results/{baseline_method}_{formatted_datetime}")
    print(f"results/{baseline_method}_{formatted_datetime}")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    train_model(baseline_method, reweight_w, sparse_data, logger, writer, data_path)


def train_model(baseline_method='dt_reweight', reweight_w=0.1, sparse_data=False, logger=None, writer=None, data_path=None, load_preprocessed_data=False):
    state_dim = 16
    logger = logger

    # 缓存文件路径
    pickle_path = f"saved_model/gas/{baseline_method}_replay_buffer{'_sparse' if sparse_data else ''}.pkl"
    os.makedirs("saved_model", exist_ok=True)

    # 1. 判断缓存pkl是否存在；有则直接读取，否则重新构造并保存
    if os.path.exists(pickle_path):
        logger.info(f"Loading replay buffer from {pickle_path}")
        with open(pickle_path, "rb") as f:
            replay_buffer = pickle.load(f)
    else:
        logger.info(f"Building replay buffer and saving to {pickle_path}")
        replay_buffer = EpisodeReplayBuffer(
            state_dim=state_dim, 
            act_dim=1, 
            data_path=data_path, 
            sparse_data=sparse_data)
        # 保存pkl
        with open(pickle_path, "wb") as f:
            pickle.dump(replay_buffer, f)
    
    # 2. 保存范数参数
    save_normalize_dict(
        {"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std},
        f"saved_model/{baseline_method}test"
    )

    logger.info(f"Replay buffer size: {len(replay_buffer.trajectories)}")

    replay_buffer.scale = 50
    
    # setup Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DecisionTransformer(state_dim=state_dim, act_dim=1, state_mean=replay_buffer.state_mean,
                                state_std=replay_buffer.state_std,
                                baseline_method=baseline_method,
                                reweight_w=reweight_w)
    model.to(device)

    critic = Critic(state_dim, 1, hidden_dim=256).to(device)  # 你的action_dim=1
    critic_target = Critic(state_dim, 1, hidden_dim=256).to(device)
    critic_target.load_state_dict(critic.state_dict())
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

    actor_optimizer = torch.optim.AdamW(model.parameters(), lr=model.learning_rate, weight_decay=model.weight_decay)

    # alpha, tau等参数可与ql_trainer类似
    alpha = 1.0 # Q正则系数，可配置
    tau = 0.005 # target_q软更新率
    discount = 0.99

    # Gradient steps and Batch size
    step_num = 16000
    batch_size = 128
    sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=step_num * batch_size, replacement=True)
    dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=batch_size)

    model.train()
    model.hyperparameters['step_num'] = step_num
    model.hyperparameters['batch_size'] = batch_size

    # Record the hyperparameters
    with open(f'results/{baseline_method}_{formatted_datetime}/model_hyperparameters.txt', 'w') as f:
        for key, value in model.hyperparameters.items():
            if isinstance(value, str):
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value}\n")

    i = 0
    for states, actions, rewards, dones, rtg, timesteps, attention_mask, ctg, score_to_go, costs in dataloader:

        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        rtg = rtg.to(device)
        timesteps = timesteps.to(device)
        attention_mask = attention_mask.to(device)
        ctg = ctg.to(device)
        score_to_go = score_to_go.to(device)
        costs = costs.to(device)

        # ================= Policy forward =================
        # forward 出action_pred:
        _, action_preds, _, _ = model.forward(states, actions, rewards, rtg[:, :-1], ctg[:, :-1], score_to_go[:, :-1], timesteps,attention_mask=attention_mask)
        
        # 维度整理
        batch_size, seq_length, act_dim = action_preds.shape
        act_pred_vec = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target_vec = actions.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        bc_loss = F.mse_loss(act_pred_vec, action_target_vec)
        
        # ================= Critic update ==================
        # TD loss只用序列内非结尾steps:
        non_final_mask = attention_mask[:, :-1].reshape(-1) > 0           # (N,)
        states_q = states[:, :-1, :].reshape(-1, states.shape[2])[non_final_mask]
        actions_q = actions[:, :-1, :].reshape(-1, actions.shape[2])[non_final_mask]
        current_q1, current_q2 = critic(states_q, actions_q)

        s_next = states[:, 1:, :].reshape(-1, states.shape[2])[non_final_mask]
        a_next = action_preds[:, 1:, :].reshape(-1, act_dim)[non_final_mask]
        with torch.no_grad():
            target_q1, target_q2 = critic_target(s_next, a_next)
            target_q = torch.min(target_q1, target_q2)
        reward_vec = rewards[:, :-1].reshape(-1, 1)[non_final_mask]
        done_vec = dones[:, :-1].reshape(-1, 1)[non_final_mask]
        td_target = reward_vec + discount * (1 - done_vec) * target_q

        critic_loss = F.mse_loss(current_q1, td_target) + F.mse_loss(current_q2, td_target)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_optimizer.step()
        
        # ======= Actor Q-regularization ==========
        # 用actor生成action_pred和环境state，action输入Critic
        all_valid_mask = attention_mask.reshape(-1) > 0
        states_all_valid = states.reshape(-1, states.shape[2])[all_valid_mask]
        act_pred_vec = action_preds.reshape(-1, action_preds.shape[2])[all_valid_mask]
        q1_new_action, q2_new_action = critic(states_all_valid, act_pred_vec)
        actor_q_loss = - torch.mean(torch.min(q1_new_action, q2_new_action))
        actor_loss = bc_loss + alpha * actor_q_loss


        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        actor_optimizer.step()

        # =========== Target network EMA ==========
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1. - tau) * target_param.data)

        if i % 1000 == 0:
            logger.info(f"Step: {i} BC loss: {bc_loss.item():.6f} Q loss: {actor_q_loss.item():.6f} Critic loss: {critic_loss.item():.6f}")
            if i % 1000 == 0:
                model.save_net(f"saved_model/gas/Q_regularized")
                critic.save_net(f"saved_model/gas/Q_regularized")
                save_normalize_dict({"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std},
                                    f"results/{baseline_method}_{formatted_datetime}/saved_model/{baseline_method}/{i}")

        # writer.add_scalar('Action loss', np.mean(train_loss), i)
        model.scheduler.step()
        i += 1

    model.save_net(f"saved_model/gas/Q_regularized")
    critic.save_net(f"saved_model/gas/Q_regularized")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training dt/cdt baselines...')

    parser.add_argument('--baseline_method', type=str, default='vanilla_dt', choices=['vanilla_dt', 'dt_reweight'], help='choose a method to run')
    parser.add_argument('--reweight_w', type=float, default=0.2, help='for dt_reweight baseline: condition = rtg + w * ctg')
    parser.add_argument('--sparse_data', type=bool, default=True, help='whether train on the AuctionNet-Sparse data')
    parser.add_argument('--data_path', type=str,default="/mnt/workspace/workgroup/zmm/AuctionNet-main/strategy_train_env/data/trajectory/autoBidding_aigb_track_final_data_trajectory_data_1.csv", help='path to load the dataset')
    # parser.add_argument('--data_path', type=str,default="/mnt/workspace/workgroup/zmm/AuctionNet-main/strategy_train_env/data/trajectory/sample_autoBidding_aigb_track_final_data_trajectory_data_1.csv", help='path to load the dataset')
    args = parser.parse_args()

    run_dt_baselines(baseline_method=args.baseline_method, reweight_w=args.reweight_w, sparse_data=args.sparse_data, data_path=args.data_path)