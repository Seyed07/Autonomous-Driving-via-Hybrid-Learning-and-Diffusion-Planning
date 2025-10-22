# IRL-DAL: Safe and Adaptive Trajectory Planning for Autonomous Driving via Energy-Guided Diffusion Models

> **Authors:** Seyed Ahmad Hosseini Miangoleh, Farzaneh Abdollahi  
> **Affiliation:** Department of Electrical Engineering, Amirkabir University of Technology (Tehran Polytechnic), Tehran, Iran  

---

## 🧠 Overview

**IRL-DAL (Inverse Reinforcement Learning with a Diffusion-based Adaptive Lookahead planner)** integrates *Imitation Learning (IL)*, *Inverse Reinforcement Learning (IRL)*, and *Reinforcement Learning (RL)* with a **Diffusion-based Safety Planner (DAL)**.  
Its objective: achieving **safe, adaptive, and human-like trajectory planning** for autonomous driving systems.

---

## 🚀 Core Contributions

1. **Hybrid IL–IRL–RL Learning Architecture**  
   - Combines **Behavioral Cloning (BC)** pre-training with **PPO fine-tuning**.  
   - Incorporates **adversarial IRL (GAIL)** for dense intent alignment.  
   - Uses a **hybrid reward** blending environment and imitation feedback.
    
2. **Diffusion-based Adaptive Lookahead (DAL) Planner**  
   - Conditional diffusion model acting as a *safety consultant*.  
   - Generates safe short-horizon trajectories using an **energy-guided** objective.  
   - Penalizes collisions, lane deviation, and control jerk dynamically.

3. **Learnable Adaptive Mask (LAM)**  
   - Lightweight perception module that adapts attention using speed & LiDAR cues.  
   - Shifts focus forward at high velocity and near obstacles when hazards are detected.  
   - Jointly optimized end-to-end with PPO.

---

## 🧩 System Architecture
<div align="center">

| Module | Description | Core Method |
|:--------|:-------------|:-------------|
| **Perception (LAM)** | Context-aware spatial attention | Learnable Adaptive Mask |
| **Policy Learning** | Hybrid IL → PPO + IRL reward | BC + PPO + GAIL |
| **Safety Planner** | Energy-guided diffusion generation | DAL Planner |
</div>
<div align="center">
<img width="1705" height="458" alt="main" src="https://github.com/user-attachments/assets/b48b8ba8-0ae4-40ba-abc9-43f193802fbe" />
<em>Figure 1 — Overview of the IRL-DAL architecture.</em>
</div>

---

## 🧮 Learning Curriculum

1. **Expert Data Generation (FSM-Aware)**  
   - Uses a deterministic **Finite State Machine (FSM)** to generate expert trajectories.  
   - Stores samples by FSM-state to preserve rare safety events.

2. **Phase 1 — Behavioral Cloning (BC)**  
   - Supervised pre-training on expert data for stable initialization.

3. **Phase 2 — PPO + Adversarial IRL Fine-tuning**  
   - On-policy refinement with hybrid rewards for smoother and safer driving.

<div align="center">
<img width="1539" height="418" alt="image" src="https://github.com/user-attachments/assets/6af32aab-c456-4dba-9758-ffa548046b3d" />
<em>Figure 3 — FSM-aware expert policy for structured data generation.</em>
</div>

---

## ⚙️ Energy-Guided Diffusion for Safety

The DAL planner minimizes a composite energy:
- \(E_{lane}\): lane adherence  
- \(E_{obs}\): obstacle avoidance  
- \(E_{jerk}\): control smoothness  

When unsafe PPO actions arise, DAL replaces them with safe alternatives and logs corrections for continual learning — forming a **self-improving safety layer**.

<div align="center">
<img width="1712" height="492" alt="image" src="https://github.com/user-attachments/assets/fe46c6b9-68f0-4bd1-ba77-4539ab55933e" />
<em>Figure 2 — Learnable Adaptive Mask (LAM) perception submodule.</em>
</div>

---

## 📈 Experimental Highlights

- **Simulator:** Webots (CARLA-style physics)  
- **Sensors:** RGB, LiDAR, and Kinematics  
- **Evaluation Metrics:** Collision rate, lateral deviation, control jerk, and success rate.  
<div align="center">
   
| Model | Mean Reward | Collision Rate ↓ | Success ↑ |
|:-------|:-------------|:----------------|:-----------|
| PPO Baseline | 85 | 0.63 | 78 % |
| Structured Replay | 120 (+41 %) | 0.30 | 88 % |
| + Generative Planner | 155 (+29 %) | 0.15 | 92 % |
| **Full IRL-DAL** | **180 (+16 %)** | **0.05** | **96 %** |

</div>
<div align="center">
  <table>
    <tr>
      <td align="center" style="vertical-align: top; padding: 10px;">
        <img src="https://github.com/user-attachments/assets/ab9d1dcf-2fcf-410e-a125-94b96bbcaea6"
             width="450" alt="Evaluation Graph"><br>
      </td>
      <td align="center" style="vertical-align: top; padding: 10px;">
        <img src="https://github.com/user-attachments/assets/999e3fcc-1cfd-4ef2-9584-d863222a1e23" 
             width="450" alt="Training Dynamics"><br>
      </td>
    </tr>
  </table>
</div>
