from abc import ABC, abstractmethod
import torch

class BaseBiddingStrategy(ABC):
    """
    Base bidding strategy interface defining methods to be implemented.
    """

    def __init__(self, budget=100, name="BaseStrategy", cpa=2, category=1):
        """
        Initialize the bidding strategy.
        parameters:
            @budget: the advertiser's budget for a delivery period.
            @cpa: the CPA constraint of the advertiser.
            @category: the index of advertiser's industry category.

        """
        self.budget = budget
        self.remaining_budget = budget
        self.name = name
        self.cpa = cpa
        self.category = category
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_device(self, device):
        """
        设置计算设备
        """
        self.device = device
        # 如果子类有模型，也需要移动到指定设备
        if hasattr(self, 'model') and self.model is not None:
            self.model = self.model.to(device)

    @abstractmethod
    def reset(self):
        """
        Reset the remaining budget to its initial state.
        Must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost, device=None):
        """
        Bids for all the opportunities in a delivery period

        parameters:
         @timeStepIndex: the index of the current decision time step.
         @pValues: the conversion action probability.
         @pValueSigmas: the prediction probability uncertainty.
         @historyPValueInfo: the history predicted value and uncertainty for each opportunity.
         @historyBid: the advertiser's history bids for each opportunity.
         @historyAuctionResult: the history auction results for each opportunity.
         @historyImpressionResult: the history impression result for each opportunity.
         @historyLeastWinningCost: the history least wining costs for each opportunity.

        return:
            Return the bids for all the opportunities in the delivery period.
        """

        pass
