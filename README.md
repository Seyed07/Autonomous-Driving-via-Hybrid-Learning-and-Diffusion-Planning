# 🚗 Advanced Autonomous Driving with Reinforcement Learning, Imitation Learning, and Diffusion Models (v2.0)

**Description:**
This project presents an advanced autonomous driving system that achieves an exceptional level of lane-following precision and obstacle avoidance safety within the Webots simulation environment. It achieves this through an innovative combination of **Reinforcement Learning (RL)** with the PPO algorithm, **Imitation Learning (IL)**, **Inverse Reinforcement Learning (IRL)**, and a novel **Diffusion Planner**. The system utilizes computer vision for lane detection and LiDAR sensors for environmental awareness, guided by a rule-based expert policy.

<p align="center">
  <img src="https://github.com/user-attachments/assets/986ae391-b70f-4203-be4b-81fb1047c9c8" width="80%" alt="ILRLOA Overview">
</p>

## Source Code Access

The source code for this project is **not publicly available** by default.
Access to the code is granted only upon **request and after approval**.

To gain access or use the code, please **contact the project lead via email** or open a **new Issue** in this repository, stating your organizational affiliation and your intended use case.

- **Email:** `seyedahmad.hosseini@aut.ac.ir` - `hosseiniahmad07@gmail.com`

Once approved, you will be granted access to the private repository. Thank you for your understanding.

---

## 🧠 Architectural Overview & Learning Framework

This project develops an autonomous driving agent that implements precise navigation in Webots by integrating **RL, IL, IRL,** and **Diffusion Models**. This multi-layered architecture allows the agent to learn from both explicit expert rules (imitation) and trial-and-error (reinforcement), while leveraging generative models to predict safe and optimal trajectories.

- **Imitation Phase:** The agent learns basic navigation skills by mimicking the expert's behavior using **Behavioral Cloning (BC)**.
- **Mixed Phase:** The agent balances **stability** (imitating the expert) and **exploration** (learning from experience) by combining environmental rewards with rewards derived from IRL.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e93aa0e7-f219-408c-91d0-6ac8be66d622" width="75%" alt="Imitation Learning Phase">
</p>

---

## 💡 Key Innovation: Diffusion Planner with Energy-Based Guidance

A standout innovation in this project is the use of a **Diffusion Planner** to generate safe and optimal trajectories (sequences of actions). This planner is based on **Denoising Diffusion Probabilistic Models (DDPMs)**.

- **Architecture:** The model employs a 1D `U-Net` to predict and remove noise from a perturbed trajectory.
- **Trajectory Generation Process:** During the reverse sampling (denoising) process, the generated trajectory is steered toward safer and more optimal regions by a **Guidance Energy function**.
- **Guidance Energy Function (`GuidanceEnergy`):** This function computes gradients from various cost components to refine the trajectory. These costs include:
    - **Lane Cost (`E_lane`):** Penalizes deviation from the lane center.
    - **LiDAR Cost (`E_lidar`):** Penalizes proximity to obstacles, with adaptive weighting based on the hazard level.
    - **Jerk Cost (`E_jerk`):** Penalizes abrupt changes in steering or speed to ensure smooth motion.
    - **Stability Cost (`E_stability`):** Encourages maintaining a stable speed and steering angle.
    - **Expert Cost (`E_expert`):** Encourages staying close to the expert's suggested trajectory.

This mechanism allows the agent to formulate a safe, short-term "plan" for the immediate future before executing an action, effectively using it as a "dynamic expert."

---

## 🎭 Imitation Learning (Behavioral Cloning)

Imitation Learning (IL) enables the agent to replicate the expert's behavior by directly mapping observations to expert actions.

### Expert Policy (Rule-Based & Vision-Based)

The expert policy generates reference actions by processing sensor data:
- **Camera-Based Lane Detection:** Extracts road lines and calculates the lane center by processing images (grayscale conversion, Canny edge detection, Hough transform).
- **LiDAR-Based Obstacle Detection:** Identifies obstacles, estimates their size, and determines an appropriate avoidance strategy by analyzing a LiDAR history matrix.
- **Finite State Machine (FSM):** Manages driving states (`LANE_FOLLOWING`, `AVOIDING`, `DRIVING_STRAIGHT`, `RETURNING`) to ensure structured and predictable behavior.

### Behavioral Cloning (BC)
- **Data Collection:** Experiences (expert observations and actions) are stored in **structured and separate experience buffers** (`state_buffers`) for each FSM state.
- **Prioritized Sampling:** Experiences are scored based on **safety priority** (proximity to obstacles), **diversity priority** (discrepancy between model and expert actions), and **phase priority**. This focuses training on critical and informative samples.
- **Training Process:** BC training is conducted using the `Adam` optimizer, `MSELoss` cost function, and a `ReduceLROnPlateau` scheduler to adjust the learning rate and prevent overfitting.

---

## 🤖 Reinforcement Learning (PPO) & Inverse Reinforcement Learning (IRL)

**Reinforcement Learning (RL)** optimizes the agent's policy through interaction with the environment to maximize cumulative reward. This project utilizes the **Proximal Policy Optimization (PPO)** algorithm.

### PPO Framework
- **Policy:** Maps observations (image and LiDAR data) to a probability distribution over continuous actions (steering and speed).
- **Feature Extractor (`CustomCNNWithLiDAR`):** A hybrid architecture that uses a CNN to process images and an MLP to process LiDAR data, producing a shared feature vector (size 512). This extractor is shared between the PPO policy and the IRL Discriminator.

### Integration with IRL
**Inverse Reinforcement Learning (IRL)** enhances the learning process by inferring a reward function from the expert's behavior.
- **Discriminator:** A classifier network that learns to distinguish between expert and agent (state-action) pairs.
- **IRL Reward:** An additional reward is computed as `$r_{irl} = -\log(1 - D(s, a))$`, where `D(s, a)` is the discriminator's predicted probability that the action `a` in state `s` is "expert-like."
- **Extractor Synchronization:** To improve stability, the weights of the Discriminator's feature extractor are periodically synchronized with the PPO's feature extractor.

### Hybrid Reward Function
The final reward is a combination of the environmental reward and the IRL reward: `$Reward = w_{env} \cdot R_{env} + w_{irl} \cdot r_{irl}$`. This structure incentivizes the agent to achieve objective goals (safety, lane-following) while also learning the expert's behavioral style.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4429ce3b-3f08-4e1a-b2eb-15720f78398f" width="80%" alt="RL IRL Combination">
</p>

---

## 📊 Results & Evaluation

The system achieved outstanding results in simulation tests. The plots below show the model's performance over time and a comparison between the agent's and expert's actions.
- **Average Reward:** Achieved a mean reward above 150 (out of a maximum of 200).
- **Collision Rate:** Less than 1 collision per 1000 timesteps.
- **Lane Following Error:** Lateral deviation of less than 15 pixels.

<!-- <p align="center">
  <img src="https://github.com/user-attachments/assets/ab9d1dcf-2fcf-410e-a125-94b96bbcaea6" width="80%" alt="Evaluation Graph">
</p> -->
<p align="center">
  <video width="80%" controls>
    <source src="https://Seyed07.github.io/Autonomous-Driving-via-Hybrid-Learning-and-Diffusion-Planning/result.mp4" type="video/mp4">
  </video>
</p>
<p align="center">
<img width="5943" height="5307" alt="5 64 50000" src="https://github.com/user-attachments/assets/0ca9e986-8939-4389-8076-ad592b68f2f5" />
</p>


---

## 🚦 Conclusion & Future Work

This project demonstrates a powerful and unified architecture combining **Imitation Learning, Reinforcement Learning, Inverse Reinforcement Learning, and Diffusion Models**. Aided by computer vision and LiDAR data, it culminates in a robust and safe autonomous driving system.

**Suggestions for Future Work:**
- **GPS/IMU Integration:** Add spatial data for large-scale navigation.
- **Sim-to-Real Transfer:** Adapt the model for use on real-world robotic platforms.
- **Transformer-Based Planner:** Replace the U-Net architecture in the Diffusion Planner with a Transformer to better capture temporal dependencies.
- **Multi-Task Learning:** Simultaneously train for additional tasks, such as traffic sign recognition.
