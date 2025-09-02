

# 🚗 ILRLOA: Autonomous Driving with Imitation and Reinforcement Learning

**Description:**  
ILRLOA is an autonomous driving system that integrates **Imitation Learning (IL)** and **Reinforcement Learning (RL)** to enable robust lane following and obstacle avoidance in a simulated environment using the Webots platform. Powered by computer vision for lane detection and LiDAR for obstacle awareness, the system leverages a vision-guided expert policy to achieve safe and efficient navigation.


![city_traffic](https://github.com/user-attachments/assets/3428dd19-cc90-4890-8ad4-00481955ac06)


## Access to Source Code

The source code for this project is **not publicly available** by default.  
Access is granted **upon request and approval**.

If you wish to access or use the code, please **contact the author via email** or **open an issue** in this repository, stating your affiliation and intended use.

- Email: `seyedahmad.hosseini@aut.ac.ir`

Once your request is approved, you will be granted access to the private repository.

Thank you for your understanding.

---

## 🧠 Overview of the Learning Framework

The ILRLOA project develops an autonomous driving system that combines **Imitation Learning (IL)** and **Reinforcement Learning (RL)** to navigate a simulated environment in Webots, achieving robust lane following and obstacle avoidance.  
The system employs a vision-based expert policy that processes camera images for lane detection and LiDAR data for obstacle awareness, enabling precise and adaptive navigation. The learning process unfolds in three phases:

-   **Imitation Phase:** The agent learns to mimic the expert’s behavior using Behavioral Cloning (BC), establishing foundational navigation skills.
-   **Mixed Phase:** The agent transitions smoothly by blending expert and RL-driven actions, balancing stability and exploration.
-   **RL Phase:** The agent fully adopts an RL policy, optimized for long-term performance and adaptability.

*This framework yields a policy that effectively balances lane-keeping precision, obstacle avoidance, and efficient progress, leveraging computer vision as a critical component for environmental understanding.*

---

## Imitation Learning (Behavioral Cloning)

**Imitation Learning (IL)** enables the agent to replicate the behavior of an expert policy by learning to map observations to expert actions. In ILRLOA, IL is implemented through **Behavioral Cloning (BC)**, where the agent is trained to mimic the expert’s steering and throttle commands.

<div align="center">
  <img src="https://github.com/user-attachments/assets/48d16375-2706-4f30-aa0a-c8fdf3712700" width="600" alt="Imitation Learning Flowchart"/>
</div>

### Expert Policy (Computer Vision-Based)

The expert policy orchestrates navigation by integrating computer vision and LiDAR data:

-   **Camera-Based Lane Detection:**  
    Processes camera images to identify lane boundaries and compute the lane center. This involves a pipeline of image preprocessing (e.g., edge detection) and geometric analysis to detect lane lines and critical features like solid yellow lines, which trigger safety overrides.

-   **LiDAR-Based Obstacle Detection:**  
    Analyzes LiDAR data to detect obstacles within predefined proximity thresholds (`MIN_SAFE_DISTANCE`, `WARNING_DISTANCE`), estimating their size and position to inform avoidance strategies.

-   **State Machine:**  
    Manages navigation states (`STATE_LANE_FOLLOWING`, `STATE_AVOIDING`, `STATE_DRIVING_STRAIGHT`, `STATE_RETURNING`) to create structured and predictable behavior.

-   **Action Generation:**  
    Produces normalized steering and throttle commands. It dynamically adjusts steering angle and straight-driving duration based on the estimated size of obstacles, ensuring adaptive maneuvers.

*This expert policy provides a robust foundation for lane following and obstacle avoidance, leveraging computer vision for precise environmental interpretation.*

### Behavioral Cloning (BC)

BC trains a neural network to replicate the expert’s actions:

-   **Data Collection:**  
    Captures observations (camera images and LiDAR data) and corresponding expert actions. It employs a **structured experience buffer** (`state_buffers`) that categorizes data based on the vehicle's state (e.g., lane following, avoiding). This ensures that training batches contain a diverse set of scenarios.

-   **Prioritized Sampling:**  
    Experiences are stored with a calculated **priority score**. This score is a function of safety (proximity to obstacles), action diversity (difference between model and expert actions), and training phase. This method prioritizes critical and informative samples, enhancing learning efficiency.

-   **Training Process:**  
    BC training is performed adaptively. In the `imitation` phase, it runs less frequently. In the `mixed` phase, its frequency increases to refine the policy with high-quality data. The training loop uses an `Adam` optimizer, `MSELoss`, and a `ReduceLROnPlateau` learning rate scheduler to prevent overfitting and adapt to performance changes.

-   **Purpose:**  
    BC accelerates learning by initializing the agent with expert-like navigation skills, reducing the need for extensive exploration in subsequent RL phases and providing a stable foundation.

---
<p align="center">
  <img src="https://github.com/user-attachments/assets/b3a41d4d-cb67-4272-9f3f-deafa52302da" width="80%" alt="Behavioral Cloning Diagram"/>
</p>

## Reinforcement Learning (PPO)

**Reinforcement Learning (RL)** optimizes the agent’s policy through interaction with the environment, maximizing cumulative rewards. ILRLOA uses **Proximal Policy Optimization (PPO)**, a stable and efficient RL algorithm, to refine navigation performance.

### PPO Framework

PPO employs a policy gradient approach:

-   **Policy:** Maps observations to a probabilistic distribution over continuous actions (steering and throttle).
-   **Value Function:** Estimates long-term rewards for state evaluation.
-   **Optimization:** Uses a clipped objective to ensure stable policy updates, balancing exploration and exploitation.

#### Environment and Rewards

-   **Observation Space:** Combines processed camera images (`(80, 160, 3)`) and LiDAR data (`180` points), providing a comprehensive view of the environment.
-   **Action Space:** Continuous steering and throttle commands, normalized for consistency.
-   **Reward Structure:** A state-aware reward function (`calculate_reward`) is used, which provides rewards tailored to the current navigation state:
    -   *Action Alignment*: Encourages similarity to expert actions during early training.
    -   *Safety*: Penalizes proximity to obstacles to prioritize collision avoidance.
    -   *Lane Keeping*: Rewards precise alignment with the lane center, penalizing deviation.
    -   *Progress*: Incentivizes efficient forward movement.

### Training Process

PPO iteratively collects environment interactions, computes advantages, and updates the policy to maximize rewards. The same neural architecture used in BC ensures continuity, enabling seamless refinement of learned behaviors.

### Obstacle Avoidance in RL

PPO enhances avoidance by optimizing steering and speed adjustments, minimizing lane deviations, and adapting to complex or unseen obstacle configurations, resulting in smoother and safer maneuvers.

---

## Combining IL and RL

The integration of IL and RL leverages the strengths of both paradigms:
-   **IL:** Provides a stable, expert-guided starting point, reducing initial exploration demands.
-   **RL:** Enhances adaptability, optimizing for complex scenarios and long-term performance.

<img width="1000" alt="IL+RL Combined Framework" src="https://github.com/user-attachments/assets/1be56b0f-dd35-4696-9a40-f3f592341042" />

### Training Phases

The system transitions through dynamically managed phases based on total timesteps (`self.total_timesteps`):

| Phase     | Duration (Timesteps)             | Description                                                                                                                              | Action Source                                                              |
| :-------- | :------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------- |
| Imitation | `imitation_duration`             | Purely mimics the expert via BC to learn foundational skills. Actions are generated solely by the expert policy.                           | `expert_action`                                                            |
| Mixed     | `mixed_duration`                 | Blends expert and RL actions for a smooth transition. The model's influence (`model_weight`) grows quadratically as this phase progresses. | `(expert_weight * expert_action) + (model_weight * model_action)`          |

### Technical Implementation Details

-   **Action Blending in Mixed Phase:** The weight of the model's action (`model_weight`) is calculated as `progress ** 1.5`, where `progress` is the normalized duration within the mixed phase. This ensures a gradual and stable transition from expert control to autonomous RL control.
-   **Deviation Control:** To prevent erratic behavior, if the L2 norm of the difference between the model's action and the expert's action exceeds a threshold (e.g., `0.5`), the model's action is dampened by blending it with the expert action (`0.3 * model_action + 0.7 * expert_action`).
-   **Structured Experience Buffering:** The system maintains separate replay buffers for different navigation states (`state_buffers`). This allows for targeted BC training on specific behaviors (e.g., obstacle avoidance). The buffers are periodically balanced (`maintain_buffer_balance`) to prevent any single state from dominating the training data.
-   **Prioritized Batch Creation:** For RL training in the `mixed` phase, batches are constructed with a high ratio (e.g., 70%) of imitation and mixed-phase experiences, sampled based on their priority. This leverages high-quality expert data to stabilize and guide the RL agent.

---

## Potential Improvements

To advance the ILRLOA framework:

-   **Enhanced Vision:** Incorporate deep learning-based lane detection (e.g., semantic segmentation) for improved robustness.
-   **Advanced IL:** Implement techniques like DAgger to iteratively refine the expert policy, reducing distribution shift.
-   **Reward Engineering:** Introduce rewards for smoother maneuvers or energy efficiency.
-   **Curriculum Learning:** Gradually increase environmental complexity to enhance training efficiency.
-   **Sensor Fusion:** Leverage advanced architectures (e.g., attention mechanisms) for better integration of vision and LiDAR data.

## 📊 Evaluation

Evaluate performance:

https://github.com/user-attachments/assets/bb417e79-160d-4497-8fbf-14d0882a6c66

---

## Advantages and Challenges

### Advantages

-   **IL:** Accelerates learning by leveraging expert knowledge, minimizing initial exploration.
-   **RL:** Enhances robustness, adapting to dynamic and unseen scenarios.
-   **Vision-Based Expert:** Provides reliable, context-aware guidance for lane following and obstacle avoidance.
-   **Combined Approach:** Balances stability (IL) with adaptability (RL) for superior performance.

### Challenges

-   **Expert Limitations:** Vision-based detection may struggle in complex or ambiguous conditions (e.g., occluded lanes).
-   **IL Distribution Shift:** Behavioral Cloning (BC) may overfit to expert actions, limiting generalization to novel scenarios.
-   **RL Efficiency:** Pure RL often converges slowly in rare or highly complex obstacle situations.
-   **Vision Sensitivity:** Lane detection algorithms can be sensitive to noise or environmental variations.

<p align="center" style="font-size:17px;">
As demonstrated in the figures above, when <b>Reinforcement Learning (RL)</b> is deployed independently—without the support of <b>Imitation Learning (IL)</b>—the system experiences a notable decline in performance, along with pronounced fluctuations in both reward acquisition and control actions.<br>
This demonstrates the necessity for further <b>tuning</b> and stabilization of the RL phase. A practical strategy to address these instabilities is to leverage the <b>experience replay buffer</b> gathered during the IL phase, enabling the RL agent to improve its performance during independent training and reduce both variance and instability.
</p>

---

## 🚦 Conclusion

ILRLOA demonstrates a sophisticated fusion of **Imitation Learning** and **Reinforcement Learning**, driven by computer vision, to achieve robust autonomous navigation. Clone, explore, and contribute to advance the future of self-driving systems
