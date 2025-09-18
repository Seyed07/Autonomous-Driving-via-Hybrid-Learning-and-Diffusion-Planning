# 🚗Autonomous Driving with Imitation Learning , Reinforcement Learning and Inverse Reinforcement Learning 

**Description:**  
ILRLOA is an advanced autonomous driving system that combines **Imitation Learning (IL)**, **Reinforcement Learning (RL)**, and **Inverse Reinforcement Learning (IRL)** to achieve robust lane following and obstacle avoidance in a Webots simulation environment. Leveraging computer vision for lane detection and LiDAR for obstacle awareness, the system employs a vision-guided expert policy to ensure safe and efficient navigation.

<p align="center">
  <img src="https://github.com/user-attachments/assets/986ae391-b70f-4203-be4b-81fb1047c9c8" width="80%" alt="ILRLOA Overview">
</p>

## Access to Source Code

The source code for ILRLOA is **not publicly available** by default.  
Access is granted **upon request and approval**.

To access or use the code, please **contact the project lead via email** or **open an issue** in the repository, stating your affiliation and intended use.

- Email: `seyedahmad.hosseini@aut.ac.ir` - `hosseiniahmad07@gmail.com`

Upon approval, you will be granted access to the private repository.

Thank you for your understanding.

---

## 🧠 Overview of the Learning Framework

The ILRLOA project develops an autonomous driving system that integrates **Imitation Learning (IL)**, **Reinforcement Learning (RL)**, and **IRL** to navigate a simulated Webots environment, achieving precise lane following and adaptive obstacle avoidance. The system uses camera-based lane detection and LiDAR-based obstacle detection, orchestrated by a vision-guided expert policy. The learning process unfolds in two main phases:

- **Imitation Phase:** The agent learns to mimic the expert’s behavior using Behavioral Cloning (BC), establishing foundational navigation skills.
- **Mixed Phase:** The agent blends expert and RL-driven actions, enhanced by IRL rewards, to balance stability and exploration.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e93aa0e7-f219-408c-91d0-6ac8be66d622" width="75%" alt="Imitation Learning Phase">
</p>

---

## Imitation Learning (Behavioral Cloning)

**Imitation Learning (IL)** enables the agent to replicate the expert policy by mapping observations to expert actions. In ILRLOA, IL is implemented through **Behavioral Cloning (BC)**, where the agent learns to mimic the expert’s steering and speed commands.

### Expert Policy (Vision-Guided)

The expert policy orchestrates navigation by processing camera and LiDAR data:

- **Camera-Based Lane Detection:**  
  Processes camera images to identify lane boundaries and compute the lane centerline. The pipeline includes preprocessing (e.g., edge detection, color filtering) and geometric analysis to detect lane lines and critical features like solid yellow lines, which trigger safety overrides.

- **LiDAR-Based Obstacle Detection:**  
  Analyzes LiDAR data to detect obstacles within predefined safety thresholds (`MIN_SAFE_DISTANCE`, `WARNING_DISTANCE`), estimating their size and position to inform avoidance strategies.

- **Finite State Machine (FSM):**  
  Manages navigation states (`LANE_FOLLOWING`, `AVOIDING`, `DRIVING_STRAIGHT`, `RETURNING`) to ensure structured and predictable behavior.

- **Action Generation:**  
  Produces normalized steering and speed commands, dynamically adjusting steering angles and straight-driving durations based on obstacle size and position for adaptive maneuvers.

*The vision-guided expert policy provides a reliable foundation for lane following and obstacle avoidance, leveraging computer vision for precise environmental interpretation.*

### Behavioral Cloning (BC)

BC trains a neural network to replicate expert actions:

- **Data Collection:**  
  Stores observations (camera images and LiDAR data) and corresponding expert actions in a **structured experience buffer** (`state_buffers`), categorized by vehicle state (e.g., lane following, avoiding). This ensures diverse training scenarios.

- **Prioritized Sampling:**  
  Experiences are assigned **priority scores** based on safety (obstacle proximity), action diversity (difference between model and expert actions), and training phase. This prioritizes critical and informative samples, enhancing learning efficiency.

- **Training Process:**  
  BC training adapts to the phase: less frequent in the `imitation` phase and more frequent in the `mixed` phase to refine the policy. It uses an `Adam` optimizer, `MSELoss`, and a `ReduceLROnPlateau` scheduler to prevent overfitting and adapt to performance changes.

- **Purpose:**  
  BC accelerates learning by initializing the agent with expert-like navigation skills, reducing exploration needs in RL phases and providing a stable foundation.

<p align="center">
  <img src="https://github.com/user-attachments/assets/32966659-7a21-4a06-b377-239d0654a955" width="75%" alt="BC Process">
</p>

---

## Reinforcement Learning (PPO)

**Reinforcement Learning (RL)** optimizes the agent’s policy through environmental interactions, maximizing cumulative rewards. ILRLOA uses **Proximal Policy Optimization (PPO)**, a stable and efficient RL algorithm, to refine navigation performance.

### PPO Framework

PPO employs a policy gradient approach:

- **Policy:** Maps observations to a probabilistic distribution over continuous actions (steering and speed).
- **Value Function:** Estimates long-term rewards for state evaluation.
- **Optimization:** Uses a clipped objective to ensure stable policy updates, balancing exploration and exploitation.

#### Environment and Rewards

- **Observation Space:** Combines resized camera images (`(64, 64, 3)`) and LiDAR data (`180` points) for a comprehensive environmental view.
- **Action Space:** Continuous steering and speed commands, normalized for consistency.
- **Reward Structure:** A state-aware reward function (`calculate_reward`) tailors rewards to navigation states:
  - *Action Alignment*: Encourages similarity to expert actions in early training.
  - *Safety*: Penalizes proximity to obstacles to prioritize collision avoidance.
  - *Lane Keeping*: Rewards alignment with the lane center, penalizing deviations.
  - *Progress*: Incentivizes efficient forward movement.

### IRL Integration

**Inverse Reinforcement Learning (IRL)** enhances RL by providing rewards based on how closely agent actions resemble expert actions. A discriminator network distinguishes expert vs. agent state-action pairs, computing rewards as `-log(1 - D(s,a))`, where `D` is the probability of being expert-like. This guides the agent toward expert-like behavior during the `mixed` phase.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4429ce3b-3f08-4e1a-b2eb-15720f78398f" width="80%" alt="RL IRL Combination">
</p>

### Training Process

PPO iteratively collects interactions, computes advantages, and updates the policy to maximize combined RL and IRL rewards. The shared neural architecture with BC ensures seamless refinement of learned behaviors.

### Obstacle Avoidance in RL

PPO, enhanced by IRL, optimizes steering and speed adjustments, minimizing lane deviations and adapting to complex or unseen obstacle configurations for smoother, safer maneuvers.

---

## 📊 Evaluation

<p align="center">
  <img src="https://github.com/user-attachments/assets/ab9d1dcf-2fcf-410e-a125-94b96bbcaea6" width="80%" alt="Evaluation Graph">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/3cc16b3c-c5dd-4d13-a989-ffee7e45fd37" width="80%" alt="Model Actions">
</p>

---

## 🚦 Conclusion

ILRLOA demonstrates a sophisticated fusion of **Imitation Learning**, **Reinforcement Learning**, and **IRL**, driven by computer vision and LiDAR, to achieve robust autonomous navigation. Clone, explore, and contribute to advance the future of self-driving systems.
