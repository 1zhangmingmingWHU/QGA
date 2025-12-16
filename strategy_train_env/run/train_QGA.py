import os
import sys
# 这两行的作用是确保无论从哪里运行脚本，都能正确导入项目模块和定位文件路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import pickle
import numpy as np
from bidding_train_env.common.utils import save_normalize_dict
from bidding_train_env.baseline.QGA.utils import EpisodeReplayBuffer
from bidding_train_env.baseline.QGA.dt_baselines import DecisionTransformer, Critic
from torch.utils.data import DataLoader, WeightedRandomSampler
import logging
import torch
import argparse
import torch.nn.functional as F
import json

def run_dt_baselines(args):
    """
    运行 Vanilla Decision Transformer 模型的训练流程。
    """
    # 1. 根据参数创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
    
    print(f"所有文件将被保存在: {args.save_dir}")

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    # 2. 调用训练函数，传递所有参数
    train_model(logger=logger, args=args)


def train_model(logger, args):
    """
    具体的模型训练函数。
    """
    # --- 1. 打印并保存所有配置参数 ---
    logger.info("=" * 30)
    logger.info("所有训练参数如下:")
    # 将参数对象转为字典，方便打印和保存
    args_dict = vars(args)
    for key, value in args_dict.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 30)
    
    # 保存参数到JSON文件，便于后续复现
    hyperparams_path = os.path.join(args.save_dir, "hyperparameters.json")
    with open(hyperparams_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    logger.info(f"超参数已保存至 {hyperparams_path}")


    # --- 2. 设置设备 ---
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，将自动切换到 CPU。")
        device = 'cpu'
    else:
        device = args.device
    logger.info(f"使用的设备: {device}")


    # --- 3. 准备数据和 Replay Buffer ---
    pickle_path = os.path.join(args.save_dir, "replay_buffer.pkl")

    if os.path.exists(pickle_path):
        logger.info(f"从 {pickle_path} 加载 Replay Buffer")
        with open(pickle_path, "rb") as f:
            replay_buffer = pickle.load(f)
    else:
        logger.info(f"创建 Replay Buffer 并保存至 {pickle_path}")
        replay_buffer = EpisodeReplayBuffer(
            state_dim=args.state_dim, 
            act_dim=args.act_dim, 
            data_path=args.data_path
        )
        with open(pickle_path, "wb") as f:
            pickle.dump(replay_buffer, f)
    
    # 保存范数参数
    norm_params_path = args.save_dir
    save_normalize_dict(
        {"state_mean": replay_buffer.state_mean.tolist(), "state_std": replay_buffer.state_std.tolist()}, norm_params_path)
    logger.info(f"范数参数已保存至 {norm_params_path}")

    logger.info(f"Replay buffer size: {len(replay_buffer.trajectories)}")
    replay_buffer.scale = args.buffer_scale
    
    # --- 4. 初始化模型和优化器 ---
    model = DecisionTransformer(
        state_dim=args.state_dim, 
        act_dim=args.act_dim, 
        state_mean=replay_buffer.state_mean,
        state_std=replay_buffer.state_std,
        # baseline_method='vanilla_dt', # 固定为vanilla_dt
        learning_rate=args.actor_lr,
        weight_decay=args.weight_decay
    )
    model.to(device)

    critic = Critic(args.state_dim, args.act_dim, hidden_dim=args.hidden_dim).to(device)
    critic_target = Critic(args.state_dim, args.act_dim, hidden_dim=args.hidden_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())
    
    actor_optimizer = torch.optim.AdamW(model.parameters(), lr=args.actor_lr, weight_decay=args.weight_decay)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    
    # --- 5. 准备数据加载器 ---
    sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=args.train_steps * args.batch_size, replacement=True)
    dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=args.batch_size)

    model.train()
    
    # --- 6. 开始训练循环 ---
    i = 0
    for states, actions, rewards, dones, rtg, timesteps, attention_mask, ctg, score_to_go, costs in dataloader:
        states, actions, rewards = states.to(device), actions.to(device), rewards.to(device)
        dones, rtg, timesteps = dones.to(device), rtg.to(device), timesteps.to(device)
        attention_mask = attention_mask.to(device)
        ctg, score_to_go, costs = ctg.to(device), score_to_go.to(device), costs.to(device)

        # Actor loss (BC + Q-value)
        _, action_preds, _, _ = model.forward(states, actions, rewards, rtg[:, :-1], ctg[:, :-1], score_to_go[:, :-1], timesteps,attention_mask=attention_mask)
        
        batch_size, seq_length, act_dim = action_preds.shape
        act_pred_vec = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target_vec = actions.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        bc_loss = F.mse_loss(act_pred_vec, action_target_vec)
        
        # Critic loss (TD error)
        non_final_mask = attention_mask[:, :-1].reshape(-1) > 0
        states_q = states[:, :-1, :].reshape(-1, args.state_dim)[non_final_mask]
        actions_q = actions[:, :-1, :].reshape(-1, args.act_dim)[non_final_mask]
        current_q1, current_q2 = critic(states_q, actions_q)

        s_next = states[:, 1:, :].reshape(-1, args.state_dim)[non_final_mask]
        a_next = action_preds[:, 1:, :].reshape(-1, act_dim)[non_final_mask]
        with torch.no_grad():
            target_q1, target_q2 = critic_target(s_next, a_next)
            target_q = torch.min(target_q1, target_q2)
        
        reward_vec = rewards[:, :-1].reshape(-1, 1)[non_final_mask]
        done_vec = dones[:, :-1].reshape(-1, 1)[non_final_mask]
        td_target = reward_vec + args.discount * (1 - done_vec) * target_q

        critic_loss = F.mse_loss(current_q1, td_target) + F.mse_loss(current_q2, td_target)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_norm_clip)
        critic_optimizer.step()
        
        # Actor Q-loss
        all_valid_mask = attention_mask.reshape(-1) > 0
        states_all_valid = states.reshape(-1, args.state_dim)[all_valid_mask]
        act_pred_vec_q_loss = action_preds.reshape(-1, act_dim)[all_valid_mask]
        q1_new_action, q2_new_action = critic(states_all_valid, act_pred_vec_q_loss)
        actor_q_loss = -torch.mean(torch.min(q1_new_action, q2_new_action))
        
        # Total actor loss
        actor_loss = bc_loss + args.alpha * actor_q_loss

        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
        actor_optimizer.step()

        # Update target network
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1. - args.tau) * target_param.data)

        # Logging and saving
        if i > 0 and i % args.log_interval == 0:
            logger.info(f"Step: {i}/{args.train_steps} | BC loss: {bc_loss.item():.4f} | Q loss: {actor_q_loss.item():.4f} | Critic loss: {critic_loss.item():.4f}")
            
            checkpoint_path_actor = os.path.join(args.save_dir, 'checkpoints', f'actor_step_{i}.pt')
            checkpoint_path_critic = os.path.join(args.save_dir, 'checkpoints', f'critic_step_{i}.pt')
            model.save_net(checkpoint_path_actor)
            critic.save_net(checkpoint_path_critic)
            logger.info(f"已保存 Checkpoint 至 {os.path.join(args.save_dir, 'checkpoints')}")

        model.scheduler.step()
        i += 1

    # --- 7. 保存最终模型 ---
    final_actor_path = os.path.join(args.save_dir, "actor_final.pt")
    final_critic_path = os.path.join(args.save_dir, "critic_final.pt")
    model.save_net(final_actor_path)
    critic.save_net(final_critic_path)
    logger.info(f"训练完成！最终模型已保存至 {final_actor_path} 和 {final_critic_path}")


def main():
    parser = argparse.ArgumentParser(description='Training Vanilla Decision Transformer...')
    
    # --- 路径和基本设置 ---
    parser.add_argument('--data_path', type=str, 
                        default="/path_to_yout_data.csv", help='数据集的路径')
    parser.add_argument('--save_dir', type=str, default='saved_model', help='模型、日志和参数的保存目录')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='训练设备 (cuda or cpu)')

    # --- 模型结构参数 ---
    parser.add_argument('--state_dim', type=int, default=16, help='状态向量的维度')
    parser.add_argument('--act_dim', type=int, default=1, help='动作向量的维度')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Critic网络隐藏层维度')

    # --- 训练过程参数 ---
    parser.add_argument('--train_steps', type=int, default=16000, help='总训练步数')
    parser.add_argument('--batch_size', type=int, default=128, help='每个训练步的批大小')
    parser.add_argument('--buffer_scale', type=int, default=50, help='Replay buffer 的缩放比例参数')
    parser.add_argument('--log_interval', type=int, default=1000, help='打印日志和保存检查点的频率')

    # --- 优化器和学习率参数 ---
    parser.add_argument('--actor_lr', type=float, default=1e-5, help='Actor (Decision Transformer) 的学习率')
    parser.add_argument('--critic_lr', type=float, default=3e-4, help='Critic 网络的学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='AdamW 优化器的权重衰减')
    parser.add_argument('--grad_norm_clip', type=float, default=1.0, help='梯度裁剪的范数阈值')

    # --- RL 算法超参数 ---
    parser.add_argument('--discount', type=float, default=0.99, help='折扣因子 (gamma)')
    parser.add_argument('--tau', type=float, default=0.005, help='目标网络软更新系数')
    parser.add_argument('--alpha', type=float, default=1.0, help='BC loss 和 Q-value loss 之间的平衡系数')

    args = parser.parse_args()

    # 调用主运行函数
    run_dt_baselines(args)


if __name__ == "__main__":
    main()
