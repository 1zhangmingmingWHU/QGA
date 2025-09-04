import time
# import gin
import numpy as np
import os
import psutil
import sys
# 添加项目根目录到 Python 路径
project_root = "/mnt/workspace/workgroup/zmm/AuctionNet-main/strategy_train_env"
sys.path.append(project_root)

# from bidding_train_env.baseline.qt.ql_DT import DecisionTransformer, Critic
from bidding_train_env.baseline.gas_dt_Q.gas_dt_Q import DecisionTransformer, Critic
from bidding_train_env.strategy.base_bidding_strategy import BaseBiddingStrategy

import torch
import pickle
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# sys.path.append('/mnt/workspace/workgroup/zmm/AuctionNet-main/strategy_train_env/bidding_train_env')

class QtBiddingStrategy(BaseBiddingStrategy):

    def __init__(self, budget=100, name="QT-PlayerStrategy", cpa=2, category=1,
                 model_name="epoch_41_eta_10_scale_1000.pth", model_param = {
    # "save_dir":"/mnt/workspace/workgroup/zmm/AuctionNet-main/strategy_train_env/saved_model/QTtest",
    "save_dir":"/mnt/workspace/workgroup/zmm/AuctionNet-main/strategy_train_env/saved_model/GAS_DT_Q_test/chusaishuju",
    "state_dim":16,
    "act_dim":1,
    "max_length":20,
    "max_ep_len":1000,
    "hidden_size":256,
    "n_layer":4,
    "n_head":4,
    "activation_function":'relu',
    "dropout":0.1,
    "scale":1000,
    "sar":False,
    "rtg_no_q":False,
    "infer_no_q":False
}):
        super().__init__(budget, name, cpa, category)
        
        file_name = os.path.dirname(os.path.realpath(__file__))
        dir_name = os.path.dirname(file_name)
        dir_name = os.path.dirname(dir_name)
        # picklePath = os.path.join(dir_name, "saved_model", "QTtest", "normalize_dict_all_data.pkl")
        picklePath = os.path.join(dir_name, "saved_model", "GAS_DT_Q_test", "chusaishuju", "normalize_dict.pkl")

        with open(picklePath, 'rb') as f:
            normalize_dict = pickle.load(f)

        self.model = DecisionTransformer(
        state_dim=model_param['state_dim'],
        act_dim=model_param['act_dim'],
        state_mean=normalize_dict["state_std"],
        state_std=normalize_dict["state_std"],
        max_length=model_param['max_length'],
        max_ep_len=model_param['max_ep_len'],
        hidden_size=model_param['hidden_size'],
        n_layer=model_param['n_layer'],
        n_head=model_param['n_head'],
        n_inner=4*model_param['hidden_size'],
        activation_function=model_param['activation_function'],
        n_positions=1024,
        resid_pdrop=model_param['dropout'],
        attn_pdrop=model_param['dropout'],
        scale=model_param['scale'],
        sar=model_param['sar'],
        rtg_no_q=model_param['rtg_no_q'],
        infer_no_q=model_param['infer_no_q']
    )
        self.critic = Critic(
            model_param['state_dim'], model_param['act_dim'] , hidden_dim=model_param['hidden_size']
        )

        checkpoint_path = os.path.join(model_param['save_dir'], model_name)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

        self.model = self.model.to(device=device)
        self.critic = self.critic.to(device=device)

    def reset(self):
        self.remaining_budget = self.budget

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        # 这个函数要改
        time_left = (48 - timeStepIndex) / 48
        budget_left = self.remaining_budget / self.budget if self.budget > 0 else 0
        history_xi = [result[:, 0] for result in historyAuctionResult]
        history_pValue = [result[:, 0] for result in historyPValueInfo]
        history_conversion = [result[:, 1] for result in historyImpressionResult]

        historical_xi_mean = np.mean([np.mean(xi) for xi in history_xi]) if history_xi else 0

        historical_conversion_mean = np.mean(
            [np.mean(reward) for reward in history_conversion]) if history_conversion else 0

        historical_LeastWinningCost_mean = np.mean(
            [np.mean(price) for price in historyLeastWinningCost]) if historyLeastWinningCost else 0

        historical_pValues_mean = np.mean([np.mean(value) for value in history_pValue]) if history_pValue else 0

        historical_bid_mean = np.mean([np.mean(bid) for bid in historyBid]) if historyBid else 0

        def mean_of_last_n_elements(history, n):
            l = len(history)
            last_n_data = history[max(0, l - n):l]
            if len(last_n_data) == 0:
                return 0
            else:
                return np.mean([np.mean(data) for data in last_n_data])

        last_three_xi_mean = mean_of_last_n_elements(history_xi, 3)
        last_three_conversion_mean = mean_of_last_n_elements(history_conversion, 3)
        last_three_LeastWinningCost_mean = mean_of_last_n_elements(historyLeastWinningCost, 3)
        last_three_pValues_mean = mean_of_last_n_elements(history_pValue, 3)
        last_three_bid_mean = mean_of_last_n_elements(historyBid, 3)

        current_pValues_mean = np.mean(pValues)
        current_pv_num = len(pValues)

        historical_pv_num_total = sum(len(bids) for bids in historyBid) if historyBid else 0
        last_three_ticks = slice(max(0, timeStepIndex - 3), timeStepIndex)
        last_three_pv_num_total = sum(
            [len(historyBid[i]) for i in range(max(0, timeStepIndex - 3), timeStepIndex)]) if historyBid else 0

        test_state = np.array([
            time_left, budget_left, historical_bid_mean, last_three_bid_mean,
            historical_LeastWinningCost_mean, historical_pValues_mean, historical_conversion_mean,
            historical_xi_mean, last_three_LeastWinningCost_mean, last_three_pValues_mean,
            last_three_conversion_mean, last_three_xi_mean,
            current_pValues_mean, current_pv_num, last_three_pv_num_total,
            historical_pv_num_total
        ])

        if timeStepIndex == 0:
            self.model.init_eval()

        # test_state = test_state.to(model_param['device'])
        # self.model.to(model_param['device'])
        alpha = self.model.take_actions(critic=self.critic, state=test_state, budget=self.budget, cpa=self.cpa,
                                        pre_reward=sum(history_conversion[-1]) if len(history_conversion) != 0 else None)
        bids = alpha * pValues
        print("bids: ", bids, alpha)
        return bids, alpha


