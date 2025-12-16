import numpy as np


class OfflineEnv:
    """
    Simulate an advertising bidding environment.
    """

    def __init__(self, min_remaining_budget: float = 0.1):
        """
        Initialize the simulation environment.
        :param min_remaining_budget: The minimum remaining budget allowed for bidding advertiser.
        """
        self.min_remaining_budget = min_remaining_budget

    def simulate_ad_bidding(self, pValues: np.ndarray,pValueSigmas: np.ndarray, bids: np.ndarray, leastWinningCosts: np.ndarray):
        """
        Simulate the advertising bidding process.

        :param pValues: Values of each pv .
        :param pValueSigmas: uncertainty of each pv .
        :param bids: Bids from the bidding advertiser.
        :param leastWinningCosts: Market prices for each pv.
        :return: Win values, costs spent, and winning status for each bid.

        pValues: 每种广告位的价值，可以理解为转化概率。
        pValueSigmas: 每种广告位价值的不确定性（标准差）。
        bids: 广告商的竞标价格。
        leastWinningCosts: 每种广告位的市场价格。
        """
        tick_status = bids >= leastWinningCosts # 判断竞标是否成功
        tick_cost = leastWinningCosts * tick_status # 计算竞标成功的成本
        values = np.random.normal(loc=pValues, scale=pValueSigmas) # 用正态分布模拟一个实际价值
        values = values*tick_status # 只有成功竞价的广告有实际价值
        tick_value = np.clip(values,0,1) # 约束到01之间
        tick_conversion = np.random.binomial(n=1, p=tick_value) # 用二项分布模拟转化，在这里实际价值越高的广告越容易转化，符合现实规律，因为广告位越好越容易转化

        return tick_value, tick_cost, tick_status,tick_conversion # 返回约束后的实际价值，竞标成功的成本，竞标状态，转化状态


def test():
    pv_values = np.array([10, 20, 30, 40, 50])
    pv_values_sigma = np.array([1, 2, 3, 4, 5])
    bids = np.array([15, 20, 35, 45, 55])
    market_prices = np.array([12, 22, 32, 42, 52])

    env = OfflineEnv()
    tick_value, tick_cost, tick_status,tick_conversion = env.simulate_ad_bidding(pv_values, bids, market_prices)

    print(f"Tick Value: {tick_value}")
    print(f"Tick Cost: {tick_cost}")
    print(f"Tick Status: {tick_status}")


if __name__ == '__main__':
    test()
