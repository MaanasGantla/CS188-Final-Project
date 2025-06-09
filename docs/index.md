---
title: COM SCI 188 Final Project
---

# Square Nut Assembly via Behavior Cloning

**Author:** Maanas Gantla
**Course:** COM SCI 188 (Spring 2025)  

## Problem Statement
The **NutAssemblySquare** environment randomizes a square nut’s position and orientation for each trial.  
This approach learns a closed-loop policy \(\pi: \mathbb R^{10}\to\mathbb R^7\)  using low-dimensional state (3D end-effector position, 3D nut position, 4D nut quaternion), reliably inserting the nut onto its peg within a 2,500 time step limit.


## Methodology
- **Data Preprocessing**  
  - Loaded 200 demos from `demos.npz` via `load_data.py`.  
  - Extracted 10D state \([e_t; p_t; q_t]\) and 7D action \(a_t\), then calculated zero-mean and unit-variance across each dimension.

- **Behavior Cloning Model**  
  - Two-layer MLP:  
    \[
      f(s) = W_2\,\mathrm{ReLU}(W_1\,s + b_1) + b_2,
      \quad W_1\in\mathbb R^{128\times10},\,W_2\in\mathbb R^{7\times128}.
    \]
  - The optimal hyperparameters used for training were: Mean Squared Error (MSE) loss, Adam (learning rate = 1e-3), batch = 256, 100 epochs, step-LR halving every 20 epochs.  
  - 90/10 train/validation split.

- **Alternative Methods Attempted**  
  - **Nearest-Neighbor:** replayed the single closest demo’s actions
  - **DMP+PID:** struggled to capture orientation changes and precise alignment


## Results
- **Behavior Cloning**: 41/50 trials → **82%** success rate.  
- **Learning Curves**:  
  - Final train loss approximated 0.0025, while validation loss approximated 0.0030.  
- **Hyperparameter Sweep**:

  | Hidden dim | Batch size | Learning rate | Success rate |
  |:----------:|:----------:|:-------------:|:------------:|
  | 64         | 128        | 1 × 10⁻³      | 68%          |
  | 64         | 256        | 1 × 10⁻³      | 70%          |
  | 128        | 128        | 1 × 10⁻³      | 75%          |
  | **128**    | **256**    | **1 × 10⁻³**  | **82%**      |
  | 128        | 256        | 5 × 10⁻⁴      | 78%          |


## Demo Video

<iframe width="560" height="315" src="https://www.youtube.com/embed/vgVSUsx_pqk" frameborder="0" allowfullscreen></iframe>

## Code & Report

- [Download full report (PDF)](../report.pdf)  
- [Source code on GitHub](https://github.com/MaanasGantla/CS188-Final-Project)
