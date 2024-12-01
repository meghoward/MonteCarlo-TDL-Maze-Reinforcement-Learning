# Reinforcement Learning Project 1

## Overview

This project implements solutions for a Maze environment modeled as a Markov Decision Process (MDP). The coursework focuses on employing **Dynamic Programming**, **Monte Carlo Reinforcement Learning**, and **Temporal Difference Reinforcement Learning** to solve the Maze. The Maze is personalized using parameters derived from the student's College ID.

The repository includes:
- A Jupyter Notebook with the implementations and experiments.
- A PDF report detailing methods, results, and analyses.

---

## Files

- `coursework_1_maze.ipynb`: The Jupyter Notebook containing code implementations for Dynamic Programming, Monte Carlo, and Temporal Difference learning agents.
- `coursework1_report.pdf`: A detailed report answering the coursework questions, including results and insights.
- `coursework1.py`: A Python script version of the implemented methods, compatible with the auto-marking system.

---

## Environment Description

The Maze environment consists of:
- **States**: Represented as a grid where some cells are obstacles, and others are terminal states with rewards.
- **Actions**: Four possible moves (north, south, east, west) with stochastic outcomes based on a success probability `p`.
- **Rewards**: Terminal states have rewards of either `500` (positive state) or `-50` (negative states), and each action incurs a reward of `-1`.

### Personalization
- `p` (success probability) and `γ` (discount factor) are derived from the CID's last two digits.
- The positive reward state varies based on the CID modulo operation.

---

## Implemented Agents

### 1. Dynamic Programming Agent
- **Method**: Policy Iteration.
- **Key Features**:
  - Exploits full knowledge of transition and reward matrices.
  - Iteratively evaluates and improves the policy until convergence.
- **Results**: Includes optimal policy and value functions under varying `γ` and `p`.

### 2. Monte Carlo Agent
- **Method**: Every-visit, On-policy with epsilon-greedy exploration.
- **Key Features**:
  - Learns directly from sampled episodes without access to transition/reward matrices.
  - Balances exploration and exploitation with tunable `ε`.
- **Results**: Includes learning curves and optimal policies for different exploration rates.

### 3. Temporal Difference Agent
- **Method**: SARSA with GLIE policy.
- **Key Features**:
  - Combines ideas of Monte Carlo and Dynamic Programming.
  - Adjusts exploration (`ε`) and learning rates (`α`) dynamically.
- **Results**: Includes optimal policies and insights on varying `α` and `ε`.

---

## How to Run

1. Clone the repository:

    git clone <repository-url>

2. Install dependencies:

    pip install -r requirements.txt

3. Run the Jupyter Notebook:

    jupyter notebook coursework_1_maze.ipynb

