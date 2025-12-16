import numpy as np
import math
import logging
import argparse
import os
import sys
import datetime
import random
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bidding_train_env.offline_eval.test_dataloader import TestDataLoader
from bidding_train_env.offline_eval.offline_env import OfflineEnv

project_root = "/path_to/strategy_train_env"



from bidding_train_env.strategy.qga_bidding_strategy import QGAStrategy

def setup_logger():
    now = datetime.datetime.now()
    log_dir = os.path.join(".", "log", now.strftime("%Y%m%d"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_{now.strftime('%H%M%S')}_QGA.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter("[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_format)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info(f"日志文件保存在: {log_file}")
    return logger


logger = None

def set_seed(seed=42):
    """
    全局固定随机数种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def getScore_neurips(reward, cpa, cpa_constraint):
    beta = 2
    penalty = 1
    if cpa > cpa_constraint:
        coef = cpa_constraint / (cpa + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward


def get_all_strategies():
    """获取所有可用的策略"""
    return {
        'QGA': QGAStrategy()
    }


def evaluate_strategy(agent, data_loader, env, keys,budget_ratio_f, test_dict):
    """
    评估单个策略在所有广告主上的性能
    
    Args:
        agent: 要评估的策略
        data_loader: 数据加载器
        env: 环境
        keys: 所有广告主的键值列表
        test_dict: 测试数据字典
        
    Returns:
        dict: 包含策略评估结果的字典
    """
    logger.info(f"正在测试策略: {agent.name}")
    
    total_reward = 0
    total_cost = 0
    total_score = 0
    
    # 只遍历指定的广告主索引
    # selected_indices = [11,16,17,20,27,29,30,34,44,46]
    selected_indices = [i for i in range(48)]
    advertiser_count = len(selected_indices)
    
    # 遍历指定的广告主
    for idx in selected_indices:
        if idx >= len(keys):
            logger.warning(f"索引 {idx} 超出广告主列表范围，跳过")
            continue
            
        key = keys[idx]
        
        # 从测试数据字典中获取当前广告主的数据
        df = test_dict[key]
        
        # 获取预算和CPA约束
        budget = df["budget"].iloc[0]*budget_ratio_f  # budget列中所有值一样，所以只取第一个
        cpa_constraint = df["CPAConstraint"].iloc[0]  # CPAConstraint列中所有值一样，所以只取第一个
        
        logger.info(f"处理广告主索引 {idx}: {key}, 预算: {budget}, CPA约束: {cpa_constraint}")
        
        # 重置环境和历史记录
        num_timeStepIndex, pValues, pValueSigmas, leastWinningCosts = data_loader.mock_data(key) 
        rewards = np.zeros(num_timeStepIndex)
        history = {
            'historyBids': [],
            'historyAuctionResult': [],
            'historyImpressionResult': [],
            'historyLeastWinningCost': [],
            'historyPValueInfo': []
        }
        
        # 重置代理的预算和CPA约束
        agent.budget = budget
        agent.cpa = cpa_constraint
        agent.reset()
        logger.info(f"广告主 {key} 设置预算: {budget}, CPA约束: {cpa_constraint}")

        for timeStep_index in range(num_timeStepIndex):
            pValue = pValues[timeStep_index] 
            pValueSigma = pValueSigmas[timeStep_index]
            leastWinningCost = leastWinningCosts[timeStep_index]

            if agent.remaining_budget < env.min_remaining_budget:
                bid = np.zeros(pValue.shape[0])
            else:
                bid,_ = agent.bidding(timeStep_index, pValue, pValueSigma, history["historyPValueInfo"],
                                    history["historyBids"],
                                    history["historyAuctionResult"], history["historyImpressionResult"],
                                    history["historyLeastWinningCost"])

            tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                        leastWinningCost)

            over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)
            while over_cost_ratio > 0:
                pv_index = np.where(tick_status == 1)[0]
                dropped_pv_index = np.random.choice(pv_index, int(math.ceil(pv_index.shape[0] * over_cost_ratio)),
                                                    replace=False)
                bid[dropped_pv_index] = 0
                tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(pValue, pValueSigma, bid,
                                                                                            leastWinningCost)
                over_cost_ratio = max((np.sum(tick_cost) - agent.remaining_budget) / (np.sum(tick_cost) + 1e-4), 0)

            agent.remaining_budget -= np.sum(tick_cost)
            rewards[timeStep_index] = np.sum(tick_conversion)
            temHistoryPValueInfo = [(pValue[i], pValueSigma[i]) for i in range(pValue.shape[0])]
            history["historyPValueInfo"].append(np.array(temHistoryPValueInfo))
            history["historyBids"].append(bid)
            history["historyLeastWinningCost"].append(leastWinningCost)
            temAuctionResult = np.array(
                [(tick_status[i], tick_status[i], tick_cost[i]) for i in range(tick_status.shape[0])])
            history["historyAuctionResult"].append(temAuctionResult)
            temImpressionResult = np.array([(tick_conversion[i], tick_conversion[i]) for i in range(pValue.shape[0])])
            history["historyImpressionResult"].append(temImpressionResult)
        
        # 计算当前广告主的结果
        advertiser_reward = np.sum(rewards)
        advertiser_cost = agent.budget - agent.remaining_budget
        advertiser_cpa = advertiser_cost / (advertiser_reward + 1e-10)
        advertiser_score = getScore_neurips(advertiser_reward, advertiser_cpa, cpa_constraint)
        
        logger.info(f'策略: {agent.name} - 广告主索引 {idx} 结果')
        logger.info(f'Reward: {advertiser_reward}')
        logger.info(f'Cost: {advertiser_cost}')
        logger.info(f'CPA: {advertiser_cpa}')
        logger.info(f'Score: {advertiser_score}')
        
        # 累加总结果
        total_reward += advertiser_reward
        total_cost += advertiser_cost
        total_score += advertiser_score
    
    # 计算平均结果
    avg_reward = total_reward / advertiser_count
    avg_cost = total_cost / advertiser_count
    avg_cpa = avg_cost / (avg_reward + 1e-10)
    avg_score = total_score / advertiser_count
    
    # 记录策略的结果
    strategy_result = {
        'name': agent.name,
        'reward': avg_reward,
        'cost': avg_cost,
        'cpa_real': avg_cpa,
        'cpa_constraint': sum([test_dict[key]["CPAConstraint"].iloc[0] for key in keys]) / advertiser_count,  # 平均CPA约束
        'score': avg_score,
        'total_reward': total_reward,
        'total_cost': total_cost,
        'total_score': total_score,
        'advertiser_count': advertiser_count
    }
    
    logger.info(f'策略: {agent.name} - 所有广告主结果汇总')
    logger.info(f'平均 Reward: {avg_reward}')
    logger.info(f'平均 Cost: {avg_cost}')
    logger.info(f'平均 CPA: {avg_cpa}')
    logger.info(f'平均 Score: {avg_score}')
    logger.info('-' * 50)
    
    return strategy_result


def run_test(args, period=7):
    """
    运行测试
    
    Args:
        args: 命令行参数
        period: 指定的period编号, 如果提供则使用该period的数据
        
    Returns:
        dict: 包含测试结果的字典
    """
    # 如果指定了period，则使用指定的period文件
    if period is not None:
        file_path = f'/path_to_your_data/trafficFinal/period-{period}.csv'
    else:
        file_path = '/path_to_your_data/trafficFinal/period-7.csv' 
    
    print(file_path)
    data_loader = TestDataLoader(file_path=file_path)
    env = OfflineEnv() 
    
    all_strategies = get_all_strategies()
    
    # 根据flag和策略名选择要运行的策略
    if args.flag == 1 and args.strategy:
        if args.strategy in all_strategies:
            strategies = [all_strategies[args.strategy]]
            logger.info(f"仅运行指定策略: {args.strategy}")
        else:
            logger.error(f"未找到指定的策略: {args.strategy}")
            logger.info(f"可用的策略: {list(all_strategies.keys())}")
            return None
    else:
        strategies = list(all_strategies.values())
        logger.info("运行所有策略")
    
    keys, test_dict = data_loader.keys, data_loader.test_dict 
    logger.info(f"数据集包含 {len(keys)} 个广告主")
    
    # 存储所有策略的结果
    all_results = {
        'reward': 0,
        'cost': 0,
        'cpa_real': 0,
        'total_score': 0,
        'strategies_results': [],
        'strategy_count': 0  # 添加策略计数器用于计算平均值
    }
    
    # 对每个策略进行测试
    for agent in strategies:
        # for budget_ratio_f in [0.5, 0.75, 1.0, 1.25, 1.5]:
        for budget_ratio_f in [1.0]:
            strategy_result = evaluate_strategy(agent, data_loader, env, keys, budget_ratio_f,test_dict)
            all_results['strategies_results'].append(strategy_result)
            
            # 累加总结果，后面会计算平均值
            all_results['reward'] += strategy_result['reward']
            all_results['cost'] += strategy_result['cost']
            all_results['total_score'] += strategy_result['score']
            all_results['strategy_count'] += 1
        
        # 计算平均值
        if all_results['strategy_count'] > 0:
            all_results['reward'] = all_results['reward'] / all_results['strategy_count']
            all_results['cost'] = all_results['cost'] / all_results['strategy_count']
            all_results['total_score'] = all_results['total_score'] / all_results['strategy_count']
            all_results['cpa_real'] = all_results['cost'] / (all_results['reward'] + 1e-10)
    
    return all_results


def run_all_period(args):
    """
    迭代运行多个流量period文件
    Args:
        args: 命令行参数
    """
    results = []  # List to store results of each period
    all_rewards = 0
    all_score = 0
    period_count = 0
    
    # 运行period-7到period-27的所有数据
    for period in range(7, 28):
        try:
            logger.info(f"开始评估 period-{period}")
            result = run_test(args, period)  
            if result:
                results.append(result)    
                all_rewards += result['reward']
                all_score += result['total_score']
                period_count += 1
            logger.info(f"结束评估 period-{period}")
        except Exception as e:
            logger.error(f"在period-{period}运行出错: {str(e)}")
    
    logger.info("所有评估结束")

    if period_count > 0:
        avg_reward = all_rewards / period_count
        avg_score = all_score / period_count
        logger.info(f"平均奖励: {avg_reward}. 平均得分: {avg_score}.")
        logger.info("=" * 50)
    else:
        logger.warning("没有成功运行的period, 无法计算平均值")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='竞价策略评估')
    parser.add_argument('--flag', type=int, default=0, help='设置为1时只运行指定策略')
    parser.add_argument('--strategy', type=str, help='指定要运行的策略名称')
    parser.add_argument('--all_period', action='store_true', help='是否运行所有period (7-27)')
    parser.add_argument('--seed_ad', type=int, default=3, help='seed')
    args = parser.parse_args()
    
    # 初始化日志记录器
    logger = setup_logger()
    set_seed(args.seed_ad)
    logger.info(f"seed : {args.seed_ad}")

    if args.all_period:
        run_all_period(args)
    else:
        run_test(args)
