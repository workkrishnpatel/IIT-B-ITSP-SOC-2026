# American Option Pricing: Binomial Trees, Neural Networks, and Reinforcement Learning

![Python](https://img.shields.io/badge/Python-3.13+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![scikit--learn](https://img.shields.io/badge/scikit--learn-latest-yellow)
![Status](https://img.shields.io/badge/Status-Academic%20Project-lightgrey)

An end to end project that prices American put options three different ways and compares them honestly on the same evaluation grid. A CRR binomial tree provides the ground truth benchmark. A supervised neural network learns to approximate that benchmark instantly. A Double DQN reinforcement learning agent learns the exercise decision itself, directly from simulated experience, without ever being told the binomial price.

The project does not just report accuracy numbers. It also documents where each method breaks down, which is arguably the more important part of a pricing project like this one.

## Table of Contents

* [Business Problem and Context](#business-problem-and-context)
* [System Architecture and Workflow](#system-architecture-and-workflow)
* [Dataset Specifications](#dataset-specifications)
* [Data Integrity and Leakage Prevention](#data-integrity-and-leakage-prevention)
* [Method Comparison and Results](#method-comparison-and-results)
* [Visualizations](#visualizations)
* [Model Risk and Limitations](#model-risk-and-limitations)
* [Project Directory Structure](#project-directory-structure)
* [Environment Setup and Installation](#environment-setup-and-installation)
* [Reproducing the Results](#reproducing-the-results)
* [Course Journey](#course-journey)
* [Future Improvements](#future-improvements)
* [Author Info](#author-info)

## Business Problem and Context

American options can be exercised at any point before expiry, not only at expiry like European options. That early exercise right makes them harder to price than a simple closed form formula can handle. This project asks the same question three separate ways: given a stock price, a strike, a maturity, a rate, and a volatility, what is the option worth today, and when should the holder actually exercise it.

The three methods used here represent three different philosophies for answering that question. The binomial tree solves it exactly through backward induction on a discretized price process. The neural network learns to imitate that exact solution through supervised regression, trading a small amount of accuracy for a large speed advantage. The reinforcement learning agent learns the exercise decision through trial and error inside a simulated market, without ever seeing the binomial answer during training, which is a fundamentally harder learning problem.

## System Architecture and Workflow

Contract parameters (S0, K, T, r, sigma)
        |
        v
CRR Binomial Pricer  ---->  ground truth price + exercise boundary
        |
        v
Neural Network Pricer  ---->  fast price approximation, validated against binomial
        |
        v
RL Environment + Double DQN Policy  ---->  learned exercise policy, evaluated by Monte Carlo
        |
        v
Unified Comparison  ---->  metrics, plots, and honest discussion of where each method fails


Every method is evaluated on the same canonical contract grid so that differences in the results come from the method itself, not from silently different inputs.

## Dataset Specifications

The neural network was trained on 25000 synthetic contracts, each labeled with its binomial price under the same CRR assumptions used everywhere else in this project. The five input features are spot price, strike, time to maturity, risk free rate, and volatility. The evaluation grid used for the final comparison covers 168 contracts spanning deep in the money through deep out of the money puts, short and long maturities, and low and high volatility regimes.

| Feature | Meaning |
|---|---|
| S0 | Current spot price |
| K | Strike price |
| T | Time to maturity in years |
| r | Risk free interest rate |
| sigma | Volatility |

## Data Integrity and Leakage Prevention

Reinforcement learning is particularly easy to get wrong through information leakage, so this project treats it as a first class concern rather than an afterthought.

* The RL state only ever contains the current time fraction, time to expiry, and moneyness. It never contains future stock prices or future payoffs.
* Rewards are only paid at the moment of exercise or at expiry, never repeatedly while holding, which rules out reward double counting.
* The environment raises an error if a step is attempted after the episode has already ended.
* The neural network is checked against the option's intrinsic value floor, since an American put should never be priced below max(K minus S, 0).
* All comparisons across binomial, neural network, and RL use the exact same contract parameters, so no method is quietly evaluated on an easier setup than another.

## Method Comparison and Results

All results below come from a single run on the reference contract S0=100, K=100, T=1.0, r=0.05, sigma=0.25.

### Neural network pricer accuracy against the binomial benchmark, across 168 contracts

| Metric | Value |
|---|---|
| Mean absolute error | 0.0299 |
| Mean bias | -0.0004 |
| Max absolute error | 0.1621 |

The bias sitting close to zero means the network is not systematically over or under pricing. The error is small relative to option prices that range from close to zero up into the mid thirties across the grid.

### RL policy value compared to baselines, 5000 Monte Carlo episodes each

| Policy | Value | Std Error | Exercise Rate | Avg Exercise Step |
|---|---|---|---|---|
| Binomial benchmark | 7.9636 | n/a | n/a | n/a |
| Always hold | 7.4265 | 0.1535 | 0 percent | n/a |
| Learned DQN | 7.0568 | 0.0921 | 98.88 percent | 46.05 |
| Random | 0.6902 | 0.0223 | 100 percent | 0.99 |
| Immediate exercise | 0.0000 | 0.0000 | 100 percent | 0.00 |

The DQN policy sits below the binomial benchmark by about 11 percent, and it also sits below the trivial always hold baseline, which is the more important comparison. A well trained American put policy should never do worse than simply refusing to exercise early, since holding to expiry is always an option available to it. This is discussed honestly in the limitations section below rather than hidden.

## Visualizations

### Neural network price against binomial benchmark price

![NN vs binomial](reports/figures/nn_vs_binomial.png)

Every point sits almost exactly on the y equals x line across the full price range from 0 to 35, which is the visual confirmation of the low MAE reported above.

### Binomial exercise boundary

![Exercise boundary](reports/figures/exercise_boundary.png)

The boundary starts around a spot of 75 early in the option's life and rises toward the strike of 100 as expiry approaches, which matches standard American put theory.

### RL policy exercise region

![Exercise region](reports/figures/exercise_region.png)

This plot is the most important diagnostic in the project, and it is not fully correct. A put option should only ever be exercised when the stock is below the strike. The learned policy instead exercises on both sides of the strike, including moneyness values above 1.0 where the payoff is zero and exercising is irrational. This is discussed in detail below.

## Model Risk and Limitations

This project is a learning exercise built on synthetic contracts and a CRR binomial benchmark under simplified assumptions. The results should be read as a demonstration of method and workflow, not as production trading advice or a calibrated market pricing system.

* The CRR binomial model assumes a simplified discrete stock process and does not account for dividends, transaction costs, or stochastic volatility.
* The neural network's ceiling is the binomial benchmark it was trained on. It cannot exceed the accuracy of its own training labels.
* The RL policy shows a genuine and reproducible limitation. It over exercises at extreme moneyness on both sides of the strike, which causes it to underperform even the always hold baseline. This was verified by comparing two independent training runs of the same architecture, both of which show the same pattern, which rules out a one off fluke or a stale plot.
* The most likely explanation is insufficient exploration of deep out of the money and deep in the money states during training, since episodes mostly start near the money and rarely wander into those extreme regions long enough for the agent to learn correct behavior there.
* RL evaluation always carries Monte Carlo noise. The standard errors reported above should be read alongside the point estimates, not ignored.
* Conclusion: the binomial pricer and the neural network pricer are both reliable for this project's scope. The RL policy is best described as a partially working prototype that captures the general shape of early exercise near the money but is not yet trustworthy at the extremes.

## Project Directory Structure

american-option-capstone/
    README.md
    requirements.txt
    run_full_pipeline.py
    src/
        contract.py
        pricing/
            payoffs.py
            black_scholes.py
            binomial.py
        data/
            contract_grid.py
        ml/
            nn_pricer.py
        rl/
            env.py
            policy.py
        evaluation/
            metrics.py
            comparison.py
            plots.py
    tests/
        test_payoffs.py
        test_binomial.py
        test_rl_env.py
    saved_models/
        american_put_pricing_model.keras
        scaler.pkl
        week8_dqn_online.keras
    data/
        synthetic_american_put_dataset.csv
    reports/
        comparison.csv
        policy_comparison.csv
        figures/
            nn_vs_binomial.png
            exercise_boundary.png
            exercise_region.png


## Environment Setup and Installation

```
pip install -r requirements.txt
```

## Reproducing the Results

```
pytest tests/ -v
python run_full_pipeline.py
```

The pipeline script writes `reports/comparison.csv`, `reports/policy_comparison.csv`, and the three figures referenced above. Every run uses fixed random seeds so the results should be close to reproducible run to run, aside from ordinary floating point noise.

## Course Journey

This capstone is the final week of a nine week course on American option pricing, moving from option theory through Black Scholes, binomial trees, supervised learning, and reinforcement learning. The full weekly progression is documented here.

| Week | Topic | Link |
|---|---|---|
| Week 1 | Basics of Options | [Week 1](https://github.com/workkrishnpatel/IIT-B-ITSP-SOC-2026/blob/main/week%201.md) |
| Week 2 | Black Scholes | [Week 2]() |
| Week 3 | Binomial Model | [Week 3](https://github.com/workkrishnpatel/IIT-B-ITSP-SOC-2026/blob/main/Week3_Krishn_Patel.ipynb) |
| Week 4 | Code the Baseline | [github.com/workkrishnpatel/IIT-B-ITSP-SOC-2026/tree/main/Week 4 IITB SOC DeepQuant](https://github.com/workkrishnpatel/IIT-B-ITSP-SOC-2026/tree/main/Week%204%20IITB%20SOC%20DeepQuant) |
| Week 5 | Intro to ML | [Week 5]() |
| Week 6 | Neural Network on Synthetic Data | [Week 6](https://github.com/workkrishnpatel/IIT-B-ITSP-SOC-2026/tree/main/Week6_ITSP%20SOC) |
| Week 7 | Reinforcement Learning Formulation | [Week 7](https://github.com/workkrishnpatel/IIT-B-ITSP-SOC-2026/tree/main/Week%207) |
| Week 8 | Train RL | [week 8](https://github.com/workkrishnpatel/IIT-B-ITSP-SOC-2026/tree/main/week%208) |
| Week 9 | Compare and Ship (this repository) | [Week 9](https://github.com/workkrishnpatel/IIT-B-ITSP-SOC-2026/tree/main/week%209) |

## Future Improvements

* Widen the RL training distribution so episodes visit deep in the money and deep out of the money states more often, which should directly address the exercise region problem documented above.
* Add Prioritized Experience Replay so rare but important exercise events are sampled more often during training.
* Extend the contract grid to include dividends and a term structure of interest rates.
* Package the comparison as a small interactive app so contract parameters can be changed and all three methods recomputed live.

## Author Info

* Author: Krishn Patel
* Institutional Affiliation: Indian Institute of Technology Bombay
* Course: American Option Pricing, Weeks 1 through 9 capstone
* GitHub Profile: [workkrishnpatel](https://github.com/workkrishnpatel)
