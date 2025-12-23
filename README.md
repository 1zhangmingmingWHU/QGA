# Q-Regularized Generative Auto-Bidding: From Suboptimal Trajectories to Optimal Policies

<p align="center">
  <img src="./README.assets/淘天集团logo.webp" alt="淘天集团logo" width="150">
</p>



<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-ComingSoon-blue" alt="Paper"></a>
  <a href="#"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.8%2B-important" alt="Python"></a>
  <a href="https://doi.org/10.5281/zenodo.18030705"><img src="https://zenodo.org/badge/1117295128.svg" alt="DOI"></a>
</p>


**QGA** (Q-Regularized Generative Auto-Bidding) is a novel framework designed to learn optimal bidding strategies from suboptimal historical data in computational advertising. Built on the [Decision Transformer](https://arxiv.org/abs/2106.01345), QGA introduces Q-value regularization and a dual-exploration mechanism, enabling both efficient policy imitation and robust offline exploration for better generalization in real-world advertising systems.



## Features

- **Q-value Regularization:** Integrates Q-value maximization with policy imitation for value-based offline learning with double Q-learning.
- **Dual Exploration:** Multi return-to-go and local action perturbation guided by Q-values; enables safe out-of-distribution (OOD) exploration.
- **Decision Transformer Backbone:** Leverages advanced sequence modeling for long-term trajectory dependency.
- **Robust Offline RL & Generative Modeling:** Outperforms RL and generative baselines in both offline & simulated environments.
- **Ready for Production:** Validated via large-scale online A/B testing on Taobao & Tmall.

## Method Overview

QGA addresses the problem of learning optimal auto-bidding strategies using only suboptimal (offline) trajectories. By augmenting the Decision Transformer with a Q-value regularization and dual policy exploration, QGA:

1. **Policy Learning:** Learns bidding policies from historical trajectories via supervised sequence modeling.
2. **Q-Regularization:** Regularizes policy learning with a double Q-network for action-value maximization.
3. **Dual Exploration (Inference):** Explores multi RTG targets and perturbed actions, selecting the highest-Q action per state.
4. **Deployment:** Achieves safe and robust OOD bidding, suitable for real-world advertising platforms.


## Benchmarks & Results

**Datasets:**
- [AuctionNet](https://github.com/alimama-tech/AuctionNet)
- AuctionNet-Sparse (real-world, low conversion scenario)

**Performance on AuctionNet (offline):**

| Method    | Score (Sparse, Budget=150%)|
|:---------:|:--------------------------:|
| BC        | 36.6 |
| DT        | 39.4 |
| GAS       | 46.5 |
| GAVE      | 47.4 |
| **QGA (Ours)** | **50.1** |

**Simulation Environment:**

| Method    | Score |
|:---------:|:-----:|
| IQL       | 6534  |
| DT        | 6920  |
| GAS       | 7454  |
| **QGA (Ours)** | **8113** |


**Large-scale Online A/B Test (Taobao Direct Express):**

- **Ad GMV:** ↑ 3.27%
- **Ad ROI:** ↑ 2.49%
- Generalizes across regular/promotion periods, stable cost discipline

## Installation

### Environment Preparation

Please use the following command to install the Python environment.

```
conda create -n your_env_name python=3.11.14 -y
conda activate your_env_name
pip install -r requirements.txt 
```

### Benchmark Preparation

We use AuctionNet as our benchmark. Please refer to https://github.com/alimama-tech/AuctionNet.

After installing AuctionNet, please place the code files from this repository in the appropriate locations.

## Usage

### Train

```
cd strategy_train_env/run
python train_QGA.py
```

### Evaluate

```
python run_evaluate.py
```

## Datasets

Please refer to https://tianchi.aliyun.com/competition/entrance/532236/customize448

## Citation

If you find our work useful, please consider citing us!

