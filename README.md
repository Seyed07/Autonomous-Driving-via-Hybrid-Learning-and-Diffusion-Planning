# 🚗 ILRLOA: Autonomous Driving with Imitation and Reinforcement Learning

**Description:**  
ILRLOA is an autonomous driving system that integrates **Imitation Learning (IL)** and **Reinforcement Learning (RL)** to enable robust lane following and obstacle avoidance in a simulated environment using the Webots platform. Powered by computer vision for lane detection and LiDAR for obstacle awareness, the system leverages a vision-guided expert policy to achieve safe and efficient navigation.


![city_traffic](https://github.com/user-attachments/assets/3428dd19-cc90-4890-8ad4-00481955ac06)

---

## 📖 Table of Contents

- [Overview of the Learning Framework](#-overview-of-the-learning-framework)
- [Imitation Learning (Behavioral Cloning)](#imitation-learning-behavioral-cloning)
- [Reinforcement Learning (PPO)](#reinforcement-learning-ppo)
- [Combining IL and RL](#combining-il-and-rl)
- [Obstacle Avoidance with Computer Vision](#obstacle-avoidance-with-computer-vision)
- [Advantages and Challenges](#advantages-and-challenges)
- [Potential Improvements](#potential-improvements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Evaluation](#-evaluation)
- [Repository Structure](#-repository-structure)
- [Diagrams](#-diagrams)
- [Contributing](#-contributing)
---

## 🧠 Overview of the Learning Framework

The ILRLOA project develops an autonomous driving system that combines **Imitation Learning (IL)** and **Reinforcement Learning (RL)** to navigate a simulated environment in Webots, achieving robust lane following and obstacle avoidance.  
The system employs a vision-based expert policy that processes camera images for lane detection and LiDAR data for obstacle awareness, enabling precise and adaptive navigation. The learning process unfolds in three phases:

- **Imitation Phase:** The agent learns to mimic the expert’s behavior using Behavioral Cloning (BC), establishing foundational navigation skills.
- **Mixed Phase:** The agent transitions smoothly by blending expert and RL-driven actions, balancing stability and exploration.
- **RL Phase:** The agent fully adopts an RL policy, optimized for long-term performance and adaptability.

*This framework yields a policy that effectively balances lane-keeping precision, obstacle avoidance, and efficient progress, leveraging computer vision as a critical component for environmental understanding.*

---

<div style="text-align: center;">
  <img src="https://github.com/user-attachments/assets/48d16375-2706-4f30-aa0a-c8fdf3712700" width="600" />
</div>

## Imitation Learning (Behavioral Cloning)

**Imitation Learning (IL)** enables the agent to replicate the behavior of an expert policy by learning to map observations to expert actions. In ILRLOA, IL is implemented through **Behavioral Cloning (BC)**, where the agent is trained to mimic the expert’s steering and throttle commands.

### Expert Policy (Computer Vision-Based)

The expert policy orchestrates navigation by integrating computer vision and LiDAR data:

- **Camera-Based Lane Detection:**  
  Processes camera images to identify lane boundaries and compute the lane center. This involves a pipeline of image preprocessing (e.g., normalization, edge detection) and geometric analysis to detect lane lines and critical features like yellow lines, which trigger specific maneuvers for safety.

- **LiDAR-Based Obstacle Detection:**  
  Analyzes LiDAR data to detect obstacles within predefined proximity thresholds, estimating their size and position to inform avoidance strategies.

- **State Machine:**  
  Manages navigation states:
  - **Lane Following:** Aligns the vehicle with the lane center using vision-derived steering.
  - **Avoiding:** Executes dynamic steering to bypass obstacles based on their detected position and size.
  - **Driving Straight:** Maintains forward motion post-avoidance to stabilize the vehicle.
  - **Returning:** Guides the vehicle back to the lane center using vision-based alignment.

- **Action Generation:**  
  Produces normalized steering and throttle commands, ensuring consistent and safe vehicle control.

*This expert policy provides a robust foundation for lane following and obstacle avoidance, leveraging computer vision for precise environmental interpretation.*

### Behavioral Cloning (BC)

BC trains a neural network to replicate the expert’s actions:

- **Data Collection:**  
  Captures observations (camera images and LiDAR data) and corresponding expert actions, prioritizing critical scenarios (e.g., close obstacles) to enhance learning efficiency.

- **Training Process:**  
  Employs a neural architecture that fuses vision and LiDAR features to predict actions, optimized via a supervised loss function (e.g., Mean Squared Error). Regular updates ensure the policy aligns closely with the expert’s behavior.

- **Purpose:**  
  BC accelerates learning by initializing the agent with expert-like navigation skills, reducing the need for extensive exploration in subsequent RL phases.

---

## Reinforcement Learning (PPO)

**Reinforcement Learning (RL)** optimizes the agent’s policy through interaction with the environment, maximizing cumulative rewards. ILRLOA uses **Proximal Policy Optimization (PPO)**, a stable and efficient RL algorithm, to refine navigation performance.

### PPO Framework

PPO employs a policy gradient approach:

- **Policy:** Maps observations to a probabilistic distribution over continuous actions (steering and throttle).
- **Value Function:** Estimates long-term rewards for state evaluation.
- **Optimization:** Uses a clipped objective to ensure stable policy updates, balancing exploration and exploitation.

#### Environment and Rewards

- **Observation Space:** Combines processed camera images and LiDAR data, providing a comprehensive view of the environment.
- **Action Space:** Continuous steering and throttle commands, normalized for consistency.
- **Reward Structure:**
  - *Action Alignment*: Encourages similarity to expert actions during early training.
  - *Safety*: Penalizes proximity to obstacles to prioritize collision avoidance.
  - *Lane Keeping*: Rewards precise alignment with the lane center.
  - *Progress*: Incentivizes efficient forward movement.

### Training Process

PPO iteratively collects environment interactions, computes advantages, and updates the policy to maximize rewards. The same neural architecture used in BC ensures continuity, enabling seamless refinement of learned behaviors.

### Obstacle Avoidance in RL

PPO enhances avoidance by optimizing steering and speed adjustments, minimizing lane deviations, and adapting to complex or unseen obstacle configurations, resulting in smoother and safer maneuvers.

---
<img width="1000" alt="download" src="https://github.com/user-attachments/assets/1be56b0f-dd35-4696-9a40-f3f592341042" />


## Combining IL and RL

The integration of IL and RL leverages the strengths of both paradigms:

- **IL:** Provides a stable, expert-guided starting point, reducing initial exploration demands.
- **RL:** Enhances adaptability, optimizing for complex scenarios and long-term performance.

### Training Phases

The system transitions through three phases:

| Phase      | Description                                                           |
|------------|----------------------------------------------------------------------|
| Imitation  | Mimics expert via BC for initial skills.                             |
| Mixed      | Blends expert and RL actions for transition.                         |
| RL         | Optimizes policy with PPO for robustness.                            |

### Role of Computer Vision

Computer vision is integral to the combined framework:

- **Expert Guidance:** Provides precise lane detection, enabling reliable lane following during IL and a foundation for RL.
- **Environmental Representation:** Supplies rich spatial context, enhancing both IL mimicry and RL adaptation.
- **Generalization:** Supports robust navigation across varied lane and obstacle configurations.

### State Machine Integration

A state machine structures navigation by:

- **IL:** Learning expert-defined state transitions (e.g., shifting to avoidance when obstacles are detected).
- **RL:** Refining these transitions for smoother execution and faster recovery.
- **Vision-Driven:** Ensuring accurate lane alignment in relevant states, enhancing overall performance.

---

## Obstacle Avoidance with Computer Vision

The vision-based expert policy is central to effective obstacle avoidance:

- **Lane Following Baseline:** Uses vision to maintain lane alignment when no obstacles are present.
- **Obstacle Detection and Response:** LiDAR triggers avoidance, while vision ensures lane awareness during maneuvers, preventing boundary violations.
- **Dynamic Maneuvers:** Adjusts steering based on obstacle characteristics, with vision guiding safe lane recovery.
- **Safety Overrides:** Detects critical lane features (e.g., yellow lines) to enforce urgent maneuvers.

### IL Contribution

BC captures expert avoidance behaviors, enabling the agent to learn reactive steering and speed adjustments based on obstacle proximity and position, with vision ensuring stable lane tracking.

### RL Enhancement

PPO optimizes avoidance by:

- Minimizing lane deviations through precise reward incentives.
- Enhancing safety by avoiding collisions in complex scenarios.
- Adapting to edge cases (e.g., multiple obstacles) not fully addressed by the expert.

**Example Scenario:**

- **No Obstacle:** Vision aligns the vehicle; IL mimics, RL reinforces.
- **Obstacle Detected:** Expert initiates avoidance; IL learns, RL optimizes for smoothness.
- **Post-Avoidance:** Vision guides lane recovery; RL ensures efficiency.

---

## Advantages and Challenges

### Advantages

- **IL:** Accelerates learning by leveraging expert knowledge, minimizing initial exploration.
- **RL:** Enhances robustness, adapting to dynamic and unseen scenarios.
- **Vision-Based Expert:** Provides reliable, context-aware guidance for lane following and obstacle avoidance.
- **Combined Approach:** Balances stability (IL) with adaptability (RL) for superior performance.

### Challenges

- **Expert Limitations:** Vision-based detection may struggle in complex or ambiguous conditions (e.g., occluded lanes).
- **IL Distribution Shift:** BC may overfit to expert actions, limiting generalization to new scenarios.
- **RL Efficiency:** Convergence may be slow for rare or complex obstacle configurations.
- **Vision Sensitivity:** Lane detection algorithms can be sensitive to environmental noise or variations.

---

## Potential Improvements

To advance the ILRLOA framework:

- **Enhanced Vision:** Incorporate deep learning-based lane detection (e.g., semantic segmentation) for improved robustness.
- **Advanced IL:** Implement techniques like DAgger to iteratively refine the expert policy, reducing distribution shift.
- **Reward Engineering:** Introduce rewards for smoother maneuvers or energy efficiency.
- **Curriculum Learning:** Gradually increase environmental complexity to enhance training efficiency.
- **Sensor Fusion:** Leverage advanced architectures (e.g., attention mechanisms) for better integration of vision and LiDAR data.

---

## 🛠 Installation

### Prerequisites

- Python: 3.8 or higher  
- Webots: R2023b or later ([download](https://cyberbotics.com/))  
- CUDA: Optional for GPU-accelerated training

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

<details>
<summary>requirements.txt</summary>

```
torch
stable-baselines3
opencv-python
numpy
matplotlib
gym
pillow
```
</details>

### Webots Setup

- Install Webots from https://cyberbotics.com/
- Ensure the Webots Python API is accessible.
- Use the provided world file in `worlds/`.

---

## 🚦 Usage

### Running Inference

Run the pre-trained model:

```bash
python main.py
```

- Loads the trained model and executes navigation in Webots, logging performance metrics.

**Simulation Details**

- **World:** Configured in `worlds/` with a vehicle equipped with camera and LiDAR sensors.
- **Outputs:** Continuous steering and throttle commands for real-time control.

---

## 🏋️ Training the Model

Train a new model:
Executes the phased training process, saving performance plots and the final model.

**Training Phases:**  

| Phase      | Description                                   |
|------------|-----------------------------------------------|
| Imitation  | Mimics expert via BC for initial skills.      |
| Mixed      | Blends expert and RL actions for transition.  |
| RL         | Optimizes policy with PPO for robustness.     |

---

## 📊 Evaluation

Evaluate performance:

Runs episodes, logging metrics like rewards, collision rates, and lane-keeping accuracy.  
Saves visualization plots for analysis.

---

## 📁 Repository Structure

```
├── main.py                     # Inference script
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
├── autonomous_driving_env.py   # Environment and logic
├── worlds/                     # Webots world files
├── reward_plots/               # Training/evaluation plots
├── models/                     # Saved models
├── images/                     # Diagrams and visualizations
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## 📈 Diagrams

- **Reference Diagram:**  
  *Example academic diagram for IL+RL architecture.*  
  ![Reference Architecture](https://pub.mdpi-res.com/electronics/electronics-14-01992/article_deploy/html/images/electronics-14-01992-g001.png?1747217076)

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a branch:  
   `git checkout -b feature/your-feature`
3. Commit changes:  
   `git commit -m "Add feature"`
4. Push:  
   `git push origin feature/your-feature`
5. Open a pull request.

---
## 🚦 Conclusion

ILRLOA demonstrates a sophisticated fusion of **Imitation Learning** and **Reinforcement Learning**, driven by computer vision, to achieve robust autonomous navigation. Clone, explore, and contribute to advance the future of self-driving systems!

