# IRL-DAL: Safe and Adaptive Trajectory Planning for Autonomous Driving via Energy-Guided Diffusion Models

> **Authors:** Seyed Ahmad Hosseini Miangoleh, Amin Jalal Aghdasian, Farzaneh Abdollahi  
> **Affiliation:** Department of Electrical Engineering, Amirkabir University of Technology (Tehran Polytechnic), Tehran, Iran

> **Paper Website:** [Click Here](https://seyed07.github.io/Autonomous-Driving-via-Hybrid-Learning-and-Diffusion-Planning/)

---

## 🧠 Overview

**IRL-DAL (Inverse Reinforcement Learning with a Diffusion-based Adaptive Lookahead planner)** presents a robust framework for autonomous driving that addresses the safety and stability limitations of end-to-end Reinforcement Learning.

Unlike standard RL approaches that struggle with sample inefficiency and sparse safety signals, IRL-DAL integrates:
1.  **Hybrid Learning:** Combining Behavioral Cloning (BC), PPO, and Adversarial IRL (GAIL).
2.  **Runtime Safety Shield:** A conditional **Diffusion Model** that intervenes only in critical states to prevent catastrophic failures during training and execution.
3.  **Adaptive Perception:** A learnable masking mechanism that mimics human-like attention shifts based on vehicle dynamics.

<div align="center">
<img width="100%" height="492" alt="main" src="https://github.com/user-attachments/assets/6aef18dc-b804-4af6-ab69-7d7d902b8fed" />
<em>Figure 1 — Overview of the IRL-DAL architecture: The RL agent learns from an FSM expert, while the Diffusion Planner acts as a safety shield, and LAM modulates visual attention.</em>
</div>

---

## 🚀 Core Contributions

### 1. Hybrid IL–IRL–RL Architecture
We propose a two-phase curriculum to solve the exploration-exploitation dilemma:
*   **Phase 1 (Initialization):** Supervised Behavioral Cloning (BC) using data from a rule-based FSM Expert.
*   **Phase 2 (Refinement):** PPO fine-tuning guided by a **Hybrid Reward System** (Dense Environmental Reward + Learned IRL Style Reward).

### 2. Diffusion-based Adaptive Lookahead (DAL) as a Safety Shield
Instead of using diffusion for all planning (which is slow), DAL acts as an **on-demand safety consultant**.
*   **Trigger:** Activates only when safety metrics (LiDAR distance, lane deviation) breach thresholds.
*   **Function:** Generates corrective trajectories using **Energy-Guided Sampling** (minimizing collision risk and jerk).
*   **Role:** Allows the agent to survive critical scenarios, enabling it to learn from the FSM expert instead of terminating the episode early.

### 3. Learnable Adaptive Mask (LAM)
A lightweight, interpretable attention module jointly optimized with the policy.
*   **Mechanism:** Dynamically weights input image regions based on **Speed** and **LiDAR Risk**.
*   **Behavior:** Contrary to standard attention which spreads out, LAM **amplifies the near-field (lower visual field)** at high speeds to ensure lateral stability and precise lane tracking.

---

## 🧩 System Modules

| Module | Role | Method |
|:--------|:-------------|:-------------|
| **Perception (LAM)** | Context-aware visual attention | Learnable Gradient Mask (Speed/Risk conditioned) |
| **Policy (Actor)** | Main driving decisions | PPO + BC Regularization |
| **Reward (Critic)** | Style & Safety feedback | Adversarial IRL (Discriminator) + Env Reward |
| **Safety (DAL)** | Runtime intervention & shielding | Energy-Guided Conditional Diffusion |

---

## ⚙️ Methodology Highlights

### SAEC: Safety-Aware Experience Correction
Standard RL fails when the agent crashes early. Our **SAEC** loop uses the Diffusion Planner to intervene *before* a crash.
*   **Execution:** The Diffusion action saves the vehicle.
*   **Learning:** The interaction is stored in the replay buffer, but the **Training Target** remains the FSM/Expert action. This teaches the PPO agent: *"This is how you should have handled the situation to avoid needing the shield."*

### Energy-Guided Diffusion
The DAL planner does not just clone behavior; it minimizes a composite energy function at inference time:
$$E_{total} = w_{lane}E_{lane} + w_{obs}E_{obs} + w_{jerk}E_{jerk}$$

<div align="center">
<img width="100%" height="597" alt="attention" src="https://github.com/user-attachments/assets/e729303d-5c95-4747-815e-5b56fcfe2181" />
<em>Figure 2 — (Left) LAM focusing on near-field lane lines during high-speed driving. (Right) Diffusion planner generating safe candidates.</em>
</div>

---

## 📈 Experimental Results

We evaluated IRL-DAL in **Webots** against strong baselines. The results demonstrate that adding the Diffusion Shield and LAM significantly reduces collision rates and improves trajectory smoothness.

**Metrics:**
*   **ADE/FDE:** Average/Final Displacement Error (Traj. Prediction Accuracy).
*   **Collision Rate:** Per 1000 timesteps.

<div align="center">
   
| Model | Mean Reward | Collision Rate ↓ | Success Rate ↑ | ADE (m) ↓ | FDE (m) ↓ |
|:-------|:-------------:|:----------------:|:-----------:|:-----------:|:-----------:|
| PPO Baseline | 85.2 | 0.63 | 78.1 % | 5.25 | 11.8 |
| + FSM Replay | 120.4 | 0.30 | 88.4 % | 4.10 | 9.5 |
| + Diffusion Shield | 155.1 | 0.15 | 92.0 % | 3.15 | 7.2 |
| **Full IRL-DAL** | **180.7** | **0.05** | **96.3 %** | **2.45** | **5.1** |

</div>

<div align="center">
  <table>
    <tr>
      <td align="center" style="vertical-align: top; padding: 10px;">
        <img src="https://github.com/user-attachments/assets/ab9d1dcf-2fcf-410e-a125-94b96bbcaea6"
             width="100%" alt="Evaluation Graph"><br>
        <em>Video 3: Trajectory Execution Results</em>
      </td>
      <td align="center" style="vertical-align: top; padding: 10px;">
        <img src="https://github.com/user-attachments/assets/7667c735-8924-4fde-b9e6-725fd6cb89a2" 
             width="100%" alt="Training Dynamics"><br>
        <em>Figure 4: Training Stability & Loss</em>
      </td>
    </tr>
  </table>
</div>

--- 

## 📚 Citation
If you use this work, please cite it as:
```bibtex
@misc{miangoleh2026irldalsafeadaptivetrajectory,
      title={IRL-DAL: Safe and Adaptive Trajectory Planning for Autonomous Driving via Energy-Guided Diffusion Models}, 
      author={Seyed Ahmad Hosseini Miangoleh and Amin Jalal Aghdasian and Farzaneh Abdollahi},
      year={2026},
      eprint={2601.23266},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2601.23266}, 
}
```
## 🏛 Acknowledgments
This work was developed at the **Department of Electrical Engineering**, Amirkabir University of Technology (Tehran Polytechnic)

---

## 📬 Contact

For technical questions or collaboration opportunities:

**Seyed Ahmad Hosseini Miangoleh**
📧 [seyedahmad.hosseini@aut.ac.ir](mailto:seyedahmad.hosseini@aut.ac.ir)

**Amin Jalal Aghdasian**
📧 [amin.aghdasian@aut.ac.ir](mailto:amin.aghdasian@aut.ac.ir)

**Dr. Farzaneh Abdollahi**
📧 [f_abdollahi@aut.ac.ir](mailto:f_abdollahi@aut.ac.ir)

---
