
import os
import cv2
import numpy as np
import random
import time
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from vehicle import Driver  # Webots vehicle driver module
from controller import Supervisor  # Webots supervisor controller
import gym
from gym import Env, spaces
from stable_baselines3 import PPO  # Proximal Policy Optimization algorithm from Stable Baselines3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor  # Base class for feature extractors in SB3
from stable_baselines3.common.callbacks import BaseCallback  # Base callback class for custom training hooks
import pickle  # For serializing Python objects (e.g., saving buffers)
import copy  # For deep copying objects
import collections  # For deque and other collection types
from itertools import islice  # For slicing iterables efficiently

# Set seed for all random number generators to ensure reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

SEED = 42  # Fixed seed value for reproducibility
set_seed(SEED)

# Initial configuration parameters
IMAGE_HEIGHT = 300  # Height of the input image from camera
IMAGE_WIDTH = 700   # Width of the input image from camera
IMAGE_HEIGHT_cnn = 64  # Resized height for CNN input
IMAGE_WIDTH_cnn = 64   # Resized width for CNN input
MAX_STEERING_ANGLE = 0.8  # Maximum steering angle in normalized units
MAX_SPEED = 20.0  # Maximum vehicle speed (m/s)
SMOOTHING_ALPHA = 0.2  # Smoothing factor for lane line tracking
MIN_LINE_LENGTH = 4  # Minimum length for detected lines in Hough transform
MAX_LINE_GAP = 100  # Maximum gap between line segments in Hough transform
MIN_SAFE_DISTANCE = 3.0  # Minimum safe distance to obstacles (m)
WARNING_DISTANCE = 3.5  # Warning distance for potential obstacles (m)
LIDAR_MATRIX_HEIGHT = 180  # Height of the LiDAR data matrix (number of rays)
LIDAR_MATRIX_WIDTH = 100  # Width of the LiDAR history matrix (time steps)
BASE_AVOIDANCE_STEERING = 0.4  # Base steering angle for obstacle avoidance
RETURN_THRESHOLD = 50  # Steps before attempting to return to lane after avoidance
BASE_STRAIGHT_DURATION = 100  # Base duration for straight driving after avoidance
LIDAR_FOV = np.pi  # Field of view for LiDAR (radians)
REDUCED_SPEED = MAX_SPEED * 0.3  # Reduced speed during avoidance maneuvers
SMOOTH_AVOIDANCE_FACTOR = 0.4  # Factor for smoothing avoidance steering changes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Suppress Intel MKL duplicate library warning

# Define vehicle state machine states
STATE_LANE_FOLLOWING = "Lane Following"  # Normal lane following behavior
STATE_AVOIDING = "Avoiding"  # Active obstacle avoidance
STATE_DRIVING_STRAIGHT = "Driving Straight"  # Straight driving post-avoidance
STATE_RETURNING = "Returning"  # Returning to original lane after avoidance

# Image processing functions (unchanged from original)
def preprocess_image(image):
    """
    Preprocess the input image for edge detection and lane line extraction.
    Applies normalization, Gaussian blur, grayscale conversion, Canny edge detection, and morphological closing.
    """
    if image is None or image.size == 0:
        print("Error: Invalid image provided to preprocess_image.")
        return None
    image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(image_norm, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed

def mask_yellow_lane(image):
    """
    Create a binary mask for yellow lane markings using HSV color space.
    Returns a mask where yellow pixels are white (255) and others are black (0).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([18, 80, 80])   # Lower HSV bound for yellow
    upper_yellow = np.array([40, 255, 255]) # Upper HSV bound for yellow
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask

def calculate_bottom_point(line):
    """
    Calculate the bottom point of a line segment (higher y-coordinate).
    Used for determining lane line base points near the vehicle's bottom.
    """
    x1, y1, x2, y2 = line
    return (x1, y1) if y1 > y2 else (x2, y2)

def group_and_select_lines(lines, width):
    """
    Group detected lines into left and right lane candidates and select the best ones.
    Uses slope and position relative to image center to classify lines.
    Returns the best left and right line tuples.
    """
    center_x = width // 2
    left_cands, right_cands = [], []
    if lines is None:
        return None, None
    for l in lines:
        x1, y1, x2, y2 = l[0]
        length = np.hypot(x2 - x1, y2 - y1)
        slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
        if abs(slope) > 100:  # Skip near-vertical lines
            continue
        bx, by = calculate_bottom_point((x1, y1, x2, y2))
        if slope < -0.2 and (x1 < center_x or x2 < center_x or abs(bx - center_x) < width * 0.1):
            left_cands.append(((x1, y1, x2, y2), bx, by, length))
        elif slope > 0.2 and (x1 > center_x or x2 > center_x or abs(bx - center_x) < width * 0.1):
            right_cands.append(((x1, y1, x2, y2), bx, by, length))
    def score(item):
        _, bx, by, length = item
        return (by, -abs(center_x - bx), length)  # Prioritize bottom, center, length
    left_line = max(left_cands, key=score)[0] if left_cands else None
    right_line = max(right_cands, key=score)[0] if right_cands else None
    return left_line, right_line

def draw_and_smooth(image, left_line, right_line, width, height, alpha=SMOOTHING_ALPHA, prev={'left': None, 'right': None}):
    """
    Draw detected lane lines on the image, apply smoothing to base points, and calculate lane center deviation.
    Updates prev dictionary with smoothed points for next frame.
    Returns the horizontal distance from image center to lane midpoint (positive = right deviation).
    """
    center_x = width // 2
    bottom_y = height - 1
    cv2.circle(image, (center_x, bottom_y), 5, (0, 0, 255), -1)  # Mark car center (red)
    cv2.putText(image, "Car Center", (center_x - 40, bottom_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    raw = {}
    raw['left'] = calculate_bottom_point(left_line) if left_line else (0, height)
    raw['right'] = calculate_bottom_point(right_line) if right_line else (width, height)
    smooth = {}
    for side in ['left', 'right']:
        if prev[side] is None:
            smooth[side] = raw[side]
        else:
            sx = int(alpha * raw[side][0] + (1 - alpha) * prev[side][0])
            sy = int(alpha * raw[side][1] + (1 - alpha) * prev[side][1])
            smooth[side] = (sx, sy)
        prev[side] = smooth[side]
        color = (0, 255, 255) if side == 'left' else (255, 255, 0)  # Cyan for left, yellow for right
        cv2.circle(image, smooth[side], 5, color, -1)
        cv2.putText(image, side.capitalize(), (smooth[side][0]+5, smooth[side][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    mx = (smooth['left'][0] + smooth['right'][0]) // 2
    my = (smooth['left'][1] + smooth['right'][1]) // 2
    cv2.circle(image, (mx, my), 5, (0, 255, 0), -1)  # Mark midpoint (green)
    cv2.putText(image, "Mid", (mx+5, my-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
    distance = mx - center_x
    return distance

def sample_from_buffer(buf, n):
    """
    Sample n experiences randomly from a buffer (deque or list).
    Handles cases where n > buffer size or empty buffer.
    """
    buf_list = list(buf)
    n = min(n, len(buf_list))
    if n == 0:
        return []
    idx = np.random.choice(len(buf_list), size=n, replace=False)
    return [buf_list[i] for i in idx]

# Discriminator class for Imitation from Observation via Reinforcement Learning (IRL)
class Discriminator(nn.Module):
    """
    Binary classifier to distinguish expert vs. agent actions given states.
    Used in IRL to provide reward signals based on how "expert-like" agent actions are.
    """
    def __init__(self, features_dim, action_dim):
        super(Discriminator, self).__init__()
        self.features_dim = features_dim
        self.action_dim = action_dim
        input_dim = features_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output probability [0,1] of being expert
        )

    def forward(self, features, action):
        x = torch.cat([features, action], dim=-1)
        return self.net(x)

class AutonomousDrivingEnv(Env):
    """
    Custom Gym environment for autonomous driving simulation in Webots.
    Integrates camera (vision-based lane following), LiDAR (obstacle detection),
    and a finite state machine for behaviors like lane following and avoidance.
    Supports imitation learning (BC) and IRL phases with prioritized experience replay.
    """
    supervisor_instance = None  # Shared supervisor instance for simulation control
    driver_instance = None     # Shared driver instance for vehicle control

    def __init__(self, car_def_name="MY_ROBOT"):  # Default Webots node name for the vehicle
        super().__init__()
        self.np_random = np.random.RandomState(SEED)  # NumPy random state for environment
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 1, (IMAGE_HEIGHT_cnn, IMAGE_WIDTH_cnn, 3), dtype=np.float32),  # Normalized RGB image
            "lidar": spaces.Box(0, 100, (LIDAR_MATRIX_HEIGHT,), dtype=np.float32)  # LiDAR ranges (capped at 100m)
        })
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)  # [steering, speed] normalized [-1,1]
        self.model = None  # SB3 PPO model (set later)
        self.expert_policy = None  # Frozen copy of policy for expert actions
        self.expert_action = np.zeros(2)  # Current expert action (steering, speed)
        self.current_phase = "imitation"  # Training phase: "imitation" or "mixed"
        self.total_timesteps = 0  # Global timestep counter
        self.phase_config = {
            "imitation_duration": 100,  # Timesteps for pure imitation phase
            "mixed_duration": 200,      # Additional timesteps for mixed phase
            "irl_weight": 0.3,            # Weight for IRL reward in mixed phase
            "env_reward_weight": 0.7,     # Weight for environment reward
            "disc_sync_interval": 1000,   # Sync discriminator features every N steps
            "disc_train_interval": 2000,  # Train discriminator every N steps
            "bc_train_interval": {"imitation": 1000, "mixed": 500}  # BC training frequency per phase
        }
        self.bc_optimizer = None  # Optimizer for behavioral cloning
        self.bc_criterion = None  # Loss function for BC (MSE)
        self.bc_losses = []  # History of BC losses
        self.state_buffers = {  # Prioritized replay buffers per state type
            STATE_LANE_FOLLOWING: collections.deque(maxlen=20000),
            STATE_AVOIDING: collections.deque(maxlen=15000),
            STATE_DRIVING_STRAIGHT: collections.deque(maxlen=10000),
            STATE_RETURNING: collections.deque(maxlen=10000)
        }
        self.max_total_buffer_size = 60000  # Cap total experiences across buffers
        self.last_distance_error = 0.0  # Previous lane center deviation
        self.prev_lines = {'left': None, 'right': None}  # Previous smoothed lane points
        self.lidar_matrix = np.full((LIDAR_MATRIX_HEIGHT, LIDAR_MATRIX_WIDTH), np.inf, dtype=np.float32)  # LiDAR history
        self.avoidance_side = None  # "left" or "right" for avoidance direction
        self.avoidance_steps = 0  # Counter for steps in avoidance state
        self.dynamic_steering = BASE_AVOIDANCE_STEERING  # Adaptive steering for avoidance
        self.dynamic_straight_duration = BASE_STRAIGHT_DURATION  # Adaptive straight drive duration
        self.original_lane_center = None  # Lane center before avoidance for returning
        self.algo_speed = MAX_SPEED / 2  # Algorithmic base speed
        self.algo_steering = 0.0  # Algorithmic base steering
        self.state = STATE_LANE_FOLLOWING  # Current FSM state
        self.pending_side_switch_steps = 0  # Steps to delay side switch in avoidance
        self.pending_switch_to = None  # Target side for pending switch
        self.action_diffs = {'steering': [], 'speed': []}  # Action difference history (unused)
        self.episode_rewards = []  # List of per-episode total rewards
        self.recent_rewards = collections.deque(maxlen=200)  # Recent episode rewards for monitoring
        self.best_performance = -np.inf  # Best episode reward seen
        self.episode_step_rewards = []  # Rewards per step in current episode
        self.discriminator = None  # IRL discriminator network
        self.disc_features_extractor = None  # Frozen features extractor for discriminator
        self.disc_optimizer = None  # Optimizer for discriminator
        self.disc_criterion = None  # Loss for discriminator (BCE)
        self.disc_features_dim = 512  # Output dimension of features extractor
        self.action_dim = 2  # Action space dimension (steering + speed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Compute device
        
        # Initialize shared Webots instances
        if AutonomousDrivingEnv.supervisor_instance is None:
            AutonomousDrivingEnv.supervisor_instance = Supervisor()
        self.supervisor = AutonomousDrivingEnv.supervisor_instance

        if AutonomousDrivingEnv.driver_instance is None:
            AutonomousDrivingEnv.driver_instance = Driver()
        self.driver = AutonomousDrivingEnv.driver_instance

        self.car_def_name = car_def_name
        self.car_node = self.supervisor.getFromDef(self.car_def_name)
        if self.car_node is None:
            print(f"Error: Could not find Automobile with DEF '{self.car_def_name}'")
        try:
            self.camera = self.supervisor.getDevice("camera")
            self.camera.enable(int(self.supervisor.getBasicTimeStep()))
        except Exception:
            print("Warning: No camera device found. Camera code will be skipped.")
            self.camera = None
        try:
            self.lidar = self.supervisor.getDevice("Sick LMS 291")
            self.lidar.enable(int(self.supervisor.getBasicTimeStep()))
            self.lidar.enablePointCloud()
        except Exception:
            print("Warning: No LiDAR device found. LiDAR code will be skipped.")
            self.lidar = None
        
        self.frame_counter = 0  # Frame counter for rendering/debugging
        self.episode_count = 0  # Episode counter
        self.start_positions = [  # Predefined starting positions and rotations for episodes
            {"translation": [-42.4526, 114.667, 0.342393], "rotation": [-0.00224826, 0.00228126, 0.999995, 1.56]},
            {"translation": [-36.4526, 119.667, 0.342393], "rotation": [-0.00224826, 0.00228126, 0.999995, 1.56]},
        ]
        self.start_index = 0
        self.current_start_position = self.start_positions[self.start_index]
        self.lidar_below_2_count = 0  # Counter for unsafe LiDAR readings per window
        self.lidar_timestep_counter = 0  # Counter for LiDAR safety window
        self.first_lidar_below_2_episode = None  # First episode with unsafe readings
        self.lidar_below_2_episodes = []  # List of episodes with unsafe readings
        self.t1 = 0  # Unused timestamp variable

    def seed(self, seed=None):
        """Set seed for environment's random state."""
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def set_model(self, model):
        """
        Configure the environment with the PPO model.
        Sets up BC optimizer, scheduler, expert policy, and discriminator.
        """
        self.model = model
        self.device = model.device
        self.bc_optimizer = torch.optim.Adam(
            self.model.policy.parameters(),
            lr=1e-4,
            weight_decay=1e-6,
            eps=1e-8
        )
        self.bc_criterion = nn.MSELoss()
        self.bc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.bc_optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2,
            verbose=True
        )
        self.expert_policy = self.create_expert_policy()
        self.create_discriminator()

    def create_expert_policy(self):
        """
        Create a frozen copy of the policy's features extractor for expert action computation.
        Freezes parameters and sets to eval mode.
        """
        expert = copy.deepcopy(self.model.policy.features_extractor)
        for param in expert.parameters():
            param.requires_grad = False
        expert.eval()
        return expert

    def create_discriminator(self):
        """
        Initialize the IRL discriminator and its components.
        Copies features extractor from policy (frozen) and sets up optimizer/loss.
        """
        try:
            self.discriminator = Discriminator(
                features_dim=self.disc_features_dim,
                action_dim=self.action_dim
            ).to(self.device)
            if hasattr(self.model.policy, 'features_extractor'):
                self.disc_features_extractor = type(self.model.policy.features_extractor)(
                    observation_space=self.observation_space,
                    features_dim=self.disc_features_dim
                ).to(self.device)
                self.disc_features_extractor.load_state_dict(
                    self.model.policy.features_extractor.state_dict()
                )
                for param in self.disc_features_extractor.parameters():
                    param.requires_grad = False
            else:
                raise ValueError("Model does not have features_extractor")
            self.disc_optimizer = torch.optim.Adam(
                self.discriminator.parameters(), 
                lr=3e-4, 
                weight_decay=1e-5
            )
            self.disc_criterion = nn.BCELoss()
        except Exception as e:
            print(f"[DISC_INIT_ERROR] Failed to create discriminator: {e}")
            import traceback
            traceback.print_exc()

    def sync_disc_features_extractor(self):
        """
        Periodically synchronize the discriminator's features extractor with the policy's.
        Ensures discriminator uses up-to-date state representations.
        """
        if self.model is None or self.disc_features_extractor is None:
            return
        try:
            with torch.no_grad():
                self.disc_features_extractor.load_state_dict(
                    self.model.policy.features_extractor.state_dict()
                )
            print(f"[SYNC] Discriminator features synced with policy at timestep {self.total_timesteps}")
        except Exception as e:
            print(f"[SYNC_ERROR] Failed to sync features: {e}")

    def sample_expert_data(self, size):
        """
        Sample expert experiences (from imitation phase) from state buffers.
        Filters for 'imitation' phase experiences only.
        """
        pure_expert_experiences = []
        for buf in self.state_buffers.values():
            pure_expert_experiences.extend([exp for exp in buf if exp.get('phase') == 'imitation'])
        if len(pure_expert_experiences) == 0:
            return []
        return sample_from_buffer(pure_expert_experiences, min(size, len(pure_expert_experiences)))

    def sample_agent_data(self, size):
        """
        Sample agent experiences (from mixed phase) from state buffers.
        Filters for 'mixed' phase experiences only.
        """
        agent_experiences = []
        for buf in self.state_buffers.values():
            agent_experiences.extend([exp for exp in buf if exp.get('phase') == 'mixed'])
        if len(agent_experiences) == 0:
            return []
        return sample_from_buffer(agent_experiences, min(size, len(agent_experiences)))

    def train_discriminator(self, batch_size=128, num_epochs=10):
        """
        Train the discriminator to classify expert vs. agent state-action pairs.
        Samples balanced batches, computes BCE loss, and updates weights.
        Prints training summary.
        """
        expert_batch = self.sample_expert_data(batch_size // 2)
        agent_batch = self.sample_agent_data(batch_size // 2)

        if len(expert_batch) < 16 or len(agent_batch) < 16:
            print("[IRL] Insufficient data for discriminator training")
            return

        try:
            expert_images_np = np.stack([exp['state']['image'] for exp in expert_batch])
            expert_states_image = torch.from_numpy(expert_images_np).float().permute(0, 3, 1, 2).to(self.device)
            expert_states_lidar = torch.from_numpy(np.stack([exp['state']['lidar'] for exp in expert_batch])).float().to(self.device)
            expert_actions = torch.from_numpy(np.stack([exp['expert_action'] for exp in expert_batch])).float().to(self.device)
            expert_state_dict = {"image": expert_states_image, "lidar": expert_states_lidar}
            
            with torch.no_grad():
                expert_features = self.disc_features_extractor(expert_state_dict)

            agent_images_np = np.stack([exp['state']['image'] for exp in agent_batch])
            agent_states_image = torch.from_numpy(agent_images_np).float().permute(0, 3, 1, 2).to(self.device)
            agent_states_lidar = torch.from_numpy(np.stack([exp['state']['lidar'] for exp in agent_batch])).float().to(self.device)
            agent_actions = torch.from_numpy(np.stack([exp['model_action'] for exp in agent_batch])).float().to(self.device)
            agent_state_dict = {"image": agent_states_image, "lidar": agent_states_lidar}

            with torch.no_grad():
                agent_features = self.disc_features_extractor(agent_state_dict)

            expert_labels = torch.ones(len(expert_batch), 1).to(self.device) * 0.9  # Near 1 for experts
            agent_labels = torch.zeros(len(agent_batch), 1).to(self.device) + 0.1  # Near 0 for agents

            self.discriminator.train()
            total_loss = 0
            for epoch in range(num_epochs):
                self.disc_optimizer.zero_grad()
                expert_preds = self.discriminator(expert_features, expert_actions)
                agent_preds = self.discriminator(agent_features, agent_actions)
                loss = self.disc_criterion(expert_preds, expert_labels) + self.disc_criterion(agent_preds, agent_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)  # Gradient clipping
                self.disc_optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / num_epochs
            print(f"[IRL] Discriminator trained | Expert samples: {len(expert_batch)} | Agent samples: {len(agent_batch)} | Avg Loss: {avg_loss:.4f}")

        except Exception as e:
            print(f"[IRL] Error during discriminator training: {e}")
            import traceback
            traceback.print_exc()

    def get_irl_reward(self, state, model_action):
        """
        Compute IRL reward: -log(1 - D(s,a)), where D is discriminator probability of expert.
        Clipped for stability. Higher reward for more expert-like actions.
        """
        if self.discriminator is None or self.disc_features_extractor is None:
            return 0.0

        try:
            image_input = state['image']
            if isinstance(image_input, np.ndarray):
                image_tensor = torch.from_numpy(image_input).float().to(self.device)
            else:
                image_tensor = image_input.clone().detach().to(self.device)

            if image_tensor.dim() == 3:
                image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)

            lidar_input = state['lidar']
            if isinstance(lidar_input, np.ndarray):
                lidar_tensor = torch.from_numpy(lidar_input).float().unsqueeze(0).to(self.device)
            else:
                lidar_tensor = lidar_input.clone().detach().unsqueeze(0).to(self.device)

            state_tensor_dict = {"image": image_tensor, "lidar": lidar_tensor}
            action_tensor = torch.from_numpy(model_action).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                self.discriminator.eval()
                self.disc_features_extractor.eval()
                features = self.disc_features_extractor(state_tensor_dict)
                D = self.discriminator(features, action_tensor)  # Already sigmoided
                reward = -torch.log(1.0 - D + 1e-8).squeeze().item()
                reward = np.clip(reward, -5, 5)  # Normalize for stability

            return reward

        except Exception as e:
            print(f"[IRL_REWARD_ERROR] Error in get_irl_reward: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def update_training_phase(self):
        """
        Update the current training phase based on total timesteps.
        Logs phase transitions and adjusts BC learning rate.
        """
        old_phase = self.current_phase
        if self.total_timesteps < self.phase_config["imitation_duration"]:
            self.current_phase = "imitation"
        else:
            self.current_phase = "mixed"
        
        if old_phase != self.current_phase:
            print(f"\n{'='*70}")
            print(f"[PHASE TRANSITION] {old_phase.upper():>10s} → {self.current_phase.upper():<10s}")
            print(f"[PHASE TRANSITION] Timestep: {self.total_timesteps:,}")
            print(f"[PHASE TRANSITION] Buffer sizes: { {k: len(v) for k, v in self.state_buffers.items()} }")
            if self.bc_optimizer is not None:
                self.update_bc_learning_rate()
            print(f"{'='*70}\n")

    def update_bc_learning_rate(self):
        """
        Adaptively adjust BC learning rate based on phase and recent performance.
        Lower LR in mixed phase if performance degrades.
        """
        if self.current_phase == "mixed":
            base_lr = 5e-5
            if len(self.episode_rewards) > 10:
                recent_performance = np.mean(self.episode_rewards[-10:])
                if recent_performance > self.best_performance:
                    self.best_performance = recent_performance
                if recent_performance < self.best_performance * 0.8:
                    base_lr = 1e-4
            for param_group in self.bc_optimizer.param_groups:
                param_group['lr'] = base_lr
        elif self.current_phase == "imitation":
            for param_group in self.bc_optimizer.param_groups:
                param_group['lr'] = 1e-4

    def get_action_for_phase(self, model_action):
        """
        Select action based on phase: pure expert in imitation, blended in mixed.
        Blending uses progress-based weighting; clips large differences for safety.
        Returns final action and metadata dict.
        """
        if self.current_phase == "imitation":
            final_action = self.expert_action.copy()
            action_info = {
                "source": "expert",
                "expert_weight": 1.0,
                "model_weight": 0.0
            }
            return final_action, action_info

        elif self.current_phase == "mixed":
            progress = max(0, min(1, (self.total_timesteps - self.phase_config["imitation_duration"]) / 
                       self.phase_config["mixed_duration"]))
            model_weight = progress ** 1.5  # Non-linear increase for model reliance
            expert_weight = 1 - model_weight
            
            action_diff = np.linalg.norm(model_action - self.expert_action)
            if action_diff > 0.5:  # Smooth large deviations
                model_action = (0.7 * model_action + 0.3 * self.expert_action)
            
            final_action = (expert_weight * self.expert_action + model_weight * model_action)
            action_info = {
                "source": "mixed",
                "expert_weight": expert_weight,
                "model_weight": model_weight,
                "progress": progress
            }
            return final_action, action_info

    def store_structured_experience(self, state, model_action, expert_action, reward_info):
        """
        Store experience in state-specific buffer with computed priority.
        Priority based on safety, phase, action diversity, and temporal factors.
        Logs buffer stats periodically.
        """
        min_lidar = np.min(state["lidar"])
        distance_error = reward_info.get('distance_error', 0.0)
        
        # Safety priority scaling
        if min_lidar < 1.5:
            safety_priority = 4.0
        elif min_lidar < 2.5:
            safety_priority = 3.0
        elif min_lidar < MIN_SAFE_DISTANCE:
            safety_priority = 2.0
        elif min_lidar < WARNING_DISTANCE:
            safety_priority = 1.5
        else:
            safety_priority = 1.0
        
        phase_priority = {"imitation": 2.5, "mixed": 1.8}[self.current_phase]
        action_diversity = np.linalg.norm(model_action - expert_action)
        diversity_priority = 1.0 + min(action_diversity, 2.0)
        temporal_priority = 1.0 + (self.total_timesteps % 10000) / 10000 * 0.2
        if self.current_phase == "mixed" and reward_info.get('phase') == "imitation":
            phase_priority *= 1.5
        total_priority = safety_priority * phase_priority * diversity_priority * temporal_priority
        
        experience = {
            "state": copy.deepcopy(state),
            "expert_action": expert_action.copy(),
            "model_action": model_action.copy(),
            "reward_info": copy.deepcopy(reward_info),
            "priority": total_priority,
            "phase": self.current_phase,
            "timestep": self.total_timesteps,
            "min_lidar": min_lidar,
            "action_diversity": action_diversity,
            "safety_level": safety_priority
        }
        
        if self.state in self.state_buffers:
            self.state_buffers[self.state].append(experience)
        
        if self.total_timesteps % 2000 == 0:
            state_counts = {state: len(buf) for state, buf in self.state_buffers.items()}
            avg_priority = np.mean([exp['priority'] for buf in self.state_buffers.values() for exp in buf] or [0])
            print(f"[BUFFER] State distribution: {state_counts}")
            print(f"[BUFFER] Average priority: {avg_priority:.2f}")

    def maintain_buffer_balance(self):
        """
        Trim buffers if total size exceeds cap, keeping highest-priority experiences.
        Ensures balanced replay without exceeding memory limits.
        """
        total_experiences = sum(len(buf) for buf in self.state_buffers.values())
        
        if total_experiences > self.max_total_buffer_size:
            for state_type, buffer in self.state_buffers.items():
                if len(buffer) > buffer.maxlen // 2:
                    sorted_buffer = sorted(buffer, key=lambda x: x['priority'], reverse=True)
                    self.state_buffers[state_type] = collections.deque(sorted_buffer[:buffer.maxlen // 2], maxlen=buffer.maxlen)


    def get_mixed_batch_for_bc(self, batch_size=64):
        """
        Sample an equal number of experiences from each buffer, regardless of buffer size.
        If a buffer has fewer experiences than the quota, sampling is done without replacement.
        """
        all_experiences = []
        num_buffers = len(self.state_buffers)
        if num_buffers == 0:
            print("[BC-BATCH] Warning: No buffers available.")
            return []

        # Quota for each buffer
        per_buffer = max(1, batch_size // num_buffers)

        for state_type, buffer in self.state_buffers.items():
            buf_list = list(buffer)
            if len(buf_list) == 0:
                continue  # Skip empty buffer
            try:
                # If priorities exist, sampling is done based on probabilities
                priorities = np.array([exp.get('priority', 1.0) for exp in buf_list])
                probabilities = priorities / priorities.sum() if priorities.sum() > 0 else None
                sample_count = min(per_buffer, len(buf_list))
                selected = np.random.choice(buf_list, size=sample_count, replace=False, p=probabilities)
                all_experiences.extend(selected)
            except Exception as e:
                print(f"[BC-BATCH] Error sampling from buffer {state_type}: {e}")
                # In case of error, fallback to simple random sampling
                selected = np.random.choice(buf_list, size=sample_count, replace=False)
                all_experiences.extend(selected)

        # If the number of samples is still less than batch_size (e.g., some buffers were smaller), refill the remaining
        if len(all_experiences) < batch_size:
            remaining = batch_size - len(all_experiences)
            flat_bufs = [exp for buffer in self.state_buffers.values() for exp in buffer]
            if flat_bufs:
                extra_samples = np.random.choice(flat_bufs, size=min(remaining, len(flat_bufs)), replace=False)
                all_experiences.extend(extra_samples)

        return all_experiences



    def should_run_bc_training(self):
        """
        Determine if BC training should run based on phase interval and adaptive performance checks.
        Increases frequency if recent performance drops.
        """
        bc_intervals = self.phase_config["bc_train_interval"]
        base_interval = bc_intervals[self.current_phase]

        if self.current_phase == "mixed" and len(self.recent_rewards) > 10:  # Changed from self.env.recent_rewards to self.recent_rewards
            recent_avg = np.mean(list(self.recent_rewards)[-10:])
            older_avg = np.mean(list(self.recent_rewards)[:-10]) if len(self.recent_rewards) > 20 else recent_avg
            if recent_avg < older_avg * 0.9:
                base_interval = base_interval // 2
                print(f"[ADAPTIVE-BC] Performance drop detected ({recent_avg:.2f} < {older_avg:.2f}), interval: {base_interval}")

        should_train = self.total_timesteps % base_interval == 0
        if should_train:
            print(f"[BC-TRAIN] Triggering BC training at timestep {self.total_timesteps} | Buffer size: {sum(len(buf) for buf in self.state_buffers.values())}")
        return should_train
        
    def train_behavioral_cloning_structured(self, epochs=2):
        """
        Perform behavioral cloning: train policy to mimic expert actions on sampled batch.
        Uses MSE loss + L2 reg, gradient clipping, and scheduler step.
        Logs loss periodically.
        """
        if not self.should_run_bc_training():
            return

        batch_size = 64
        batch = self.get_mixed_batch_for_bc(batch_size)

        if len(batch) < batch_size // 2:
            return

        try:
            states_image_np = np.array([exp['state']['image'] for exp in batch])
            states_lidar_np = np.array([exp['state']['lidar'] for exp in batch])
            expert_actions_np = np.array([exp['expert_action'] for exp in batch])

            states_image = torch.from_numpy(states_image_np).float().to(self.device)
            states_lidar = torch.from_numpy(states_lidar_np).float().to(self.device)
            expert_actions = torch.from_numpy(expert_actions_np).float().to(self.device)

            states_image = states_image.permute(0, 3, 1, 2)  # To CHW format

            self.model.policy.train()
            total_loss = 0
            for epoch in range(epochs):
                self.bc_optimizer.zero_grad()
                state_dict = {"image": states_image, "lidar": states_lidar}
                features = self.safe_features_extraction(self.model.policy.features_extractor, state_dict)
                latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
                action_pred = self.model.policy.action_net(latent_pi)
                loss = self.bc_criterion(action_pred, expert_actions)
                l2_reg = sum(torch.norm(p) for p in self.model.policy.parameters())
                loss += 1e-6 * l2_reg  # L2 regularization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), max_norm=1.0)
                self.bc_optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / epochs
            self.bc_losses.append(avg_loss)
            if hasattr(self, 'bc_scheduler'):
                self.bc_scheduler.step(avg_loss)

            if self.total_timesteps % 5000 < 500:
                print(f"[BC] Timestep: {self.total_timesteps:6d} | Loss: {avg_loss:.6f} | Batch: {len(states_image):3d} | Phase: {self.current_phase:>9s} | LR: {self.bc_optimizer.param_groups[0]['lr']:.1e}")

            self.model.policy.eval()

        except Exception as e:
            print(f"[BC] CRITICAL Error during training: {e}")
            import traceback
            traceback.print_exc()

    def step(self, action):
        """
        Environment step: execute action, simulate, compute rewards, store experience, train components.
        Handles phase logic, safety checks, episode termination.
        Returns next_state, reward, done, info.
        """
        self.total_timesteps += 1
        self.update_training_phase()
        
        state = self._get_state()
        self.expert_action = self.vision_based_action(state)
        model_action = action.copy()
        
        final_action, action_info = self.get_action_for_phase(model_action)
        
        actual_speed = np.clip((final_action[1] + 1.0) * MAX_SPEED / 2, 0, MAX_SPEED)
        actual_steering = np.clip(final_action[0] * MAX_STEERING_ANGLE, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
        
        self.driver.setSteeringAngle(actual_steering)
        self.driver.setCruisingSpeed(actual_speed)
        self.supervisor.step(int(self.supervisor.getBasicTimeStep()))
        self.frame_counter += 1
        
        next_state = self._get_state()
        distance = 0.0
        collision = False
        
        # Vision processing for lane detection
        if self.camera:
            img_buf = self.camera.getImage()
            if img_buf is not None:
                img = np.frombuffer(img_buf, dtype=np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
                image_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                crop = image_rgb[IMAGE_HEIGHT-110:IMAGE_HEIGHT, :]
                pre = preprocess_image(crop)
                if pre is not None:
                    lines = cv2.HoughLinesP(pre, 1, np.pi/60, 15,
                                            minLineLength=MIN_LINE_LENGTH,
                                            maxLineGap=MAX_LINE_GAP)
                    left_line, right_line = group_and_select_lines(lines, IMAGE_WIDTH)
                    distance = draw_and_smooth(crop.copy(), left_line, right_line,
                                            IMAGE_WIDTH, 110,
                                            alpha=SMOOTHING_ALPHA,
                                            prev=self.prev_lines)
        
        # LiDAR processing
        lidar_data = np.zeros(LIDAR_MATRIX_HEIGHT, dtype=np.float32)
        if self.lidar:
            lidar_data = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        lidar_data[np.isinf(lidar_data)] = 100.0
        self.lidar_matrix = self.update_lidar_matrix(lidar_data)
        min_distance = np.min(lidar_data)
        self.last_distance_error = distance
        
        if min_distance < 0.5:
            collision = True
        
        # Compute rewards
        env_reward = self.calculate_reward(
            distance_error=distance,
            min_lidar_distance=min_distance,
            collision=collision,
            current_state=self.state
        )
        
        irl_reward = 0.0
        if self.current_phase == "mixed" and self.discriminator is not None:
            irl_reward = self.get_irl_reward(state, model_action)
            # irl_reward = self.get_irl_reward(state, final_action)
        
        env_w = self.phase_config["env_reward_weight"]
        irl_w = self.phase_config["irl_weight"]
        total_reward = env_w * env_reward + irl_w * irl_reward
        
        # Penalty adjustments
        if collision:
            total_reward -= 100
        elif min_distance < 1.0:
            total_reward -= 20
        elif min_distance < 1.5:
            total_reward -= 5
        total_reward = np.clip(total_reward, -100, 100)
        
        # Fix negative rewards in returning state
        if self.state == STATE_RETURNING and total_reward < 0:
            print(f"⚠️ Fixed negative reward in RETURNING: {total_reward} -> 10")
            total_reward = 10
        
        self.episode_step_rewards.append(total_reward)
        
        # Calculate action similarity and safety score based on final_action
        action_similarity = np.dot(final_action, self.expert_action) / (np.linalg.norm(final_action) * np.linalg.norm(self.expert_action) + 1e-6) if np.linalg.norm(final_action) * np.linalg.norm(self.expert_action) > 0 else 0.0
        safety_score = 1.0 if min_distance > MIN_SAFE_DISTANCE else max(0.0, min_distance / MIN_SAFE_DISTANCE)
        
        reward_info = {
            'state_reward': env_reward,
            'irl_reward': irl_reward,
            'total': total_reward,
            'state': self.state,
            'distance_error': distance,
            'min_lidar_distance': min_distance,
            'collision': collision,
            'phase': self.current_phase,
            'similarity': action_similarity,  # Now based on final_action
            'safety': safety_score
        }
        
        full_reward_info = reward_info.copy()
        full_reward_info["action_info"] = action_info
        full_reward_info["timestep"] = self.total_timesteps
        self.store_structured_experience(next_state, model_action, self.expert_action, full_reward_info)
        
        # Periodic training
        if self.total_timesteps % self.phase_config["disc_train_interval"] == 0:
            self.maintain_buffer_balance()
            self.train_discriminator(batch_size=128, num_epochs=5)
        
        if self.total_timesteps % self.phase_config["disc_sync_interval"] == 0:
            self.sync_disc_features_extractor()
        
        if self.should_run_bc_training():
            self.train_behavioral_cloning_structured(epochs=2)
        
        done = False
        # Safety termination check
        if min_distance < 0.7:
            self.lidar_below_2_count += 1
            if self.first_lidar_below_2_episode is None:
                self.first_lidar_below_2_episode = self.episode_count
            self.lidar_below_2_episodes.append(self.episode_count)
        
        self.lidar_timestep_counter += 1
        if self.lidar_timestep_counter >= 20:
            if self.lidar_below_2_count >= 10:
                done = True
                total_reward = -100
                print("[EPISODE END] Too many dangerous situations")
            self.lidar_below_2_count = 0
            self.lidar_timestep_counter = 0
        
        # Goal check
        car_position = self.car_node.getField("translation").getSFVec3f()
        if car_position[0] > 83:
            done = True
            total_reward = 200
            print("[EPISODE END] Target reached successfully!")
        
        # Episode reward aggregation
        if done:
            episode_reward = sum(self.episode_step_rewards)
            self.episode_rewards.append(episode_reward)
            self.recent_rewards.append(episode_reward)
            
            if episode_reward > self.best_performance:
                self.best_performance = episode_reward
            self.episode_step_rewards = []
        
        info = {
            'reward_info': reward_info,
            'action_info': action_info,
            'phase': self.current_phase,
            'min_distance': min_distance,
            'lane_distance': distance,
            'collision': collision,
            'current_state': self.state,
            'buffer_sizes': {k: len(v) for k, v in self.state_buffers.items()},
            'expert_action': self.expert_action.copy(),
            'final_action': final_action.copy(),
            'total_timesteps': self.total_timesteps
        }
        
        # Periodic logging
        if self.total_timesteps % 100 == 0:
            print(f"[T{self.total_timesteps:6d}] Phase: {self.current_phase:10s} | "
                f"State: {self.state:15s} | Action: {action_info['source']:12s} | "
                f"EnvR: {env_reward:.2f} | IRLR: {irl_reward:.2f} | TotalR: {total_reward:.2f}")

        return next_state, total_reward, done, info

    def reset(self):
        """
        Reset environment: increment episode, reset state vars, reposition vehicle.
        Logs episode summary.
        Returns initial observation.
        """
        self.episode_count += 1
        self.frame_counter = 0
        self.last_distance_error = 0.0
        self.prev_lines = {'left': None, 'right': None}
        self.lidar_matrix = np.full((LIDAR_MATRIX_HEIGHT, LIDAR_MATRIX_WIDTH), np.inf, dtype=np.float32)
        self.avoidance_side = None
        self.avoidance_steps = 0
        self.state = STATE_LANE_FOLLOWING
        self.pending_side_switch_steps = 0
        self.pending_switch_to = None
        self.episode_step_rewards = []
        
        if self.episode_count % 1 == 0:
            print(f"\n[EPISODE] Episode {self.episode_count} completed")
            print(f"[EPISODE] Total timesteps: {self.total_timesteps:,}")
            print(f"[EPISODE] Current phase: {self.current_phase}")
            print(f"[EPISODE] Buffer sizes: { {k: len(v) for k, v in self.state_buffers.items()} }")
            if self.bc_losses:
                recent_bc_loss = np.mean(self.bc_losses[-20:]) if len(self.bc_losses) >= 20 else np.mean(self.bc_losses)
                print(f"[EPISODE] Recent BC loss: {recent_bc_loss:.6f}")
        
        if self.car_node:
            self.start_index = (self.start_index + 1) % len(self.start_positions)
            self.current_start_position = self.start_positions[self.start_index]
            self.car_node.getField("translation").setSFVec3f(self.current_start_position["translation"])
            self.car_node.getField("rotation").setSFRotation(self.current_start_position["rotation"])
            self.car_node.resetPhysics()
        
        self.supervisor.step(int(self.supervisor.getBasicTimeStep()))
        return self._get_state()

    def update_lidar_matrix(self, lidar_data):
        """
        Shift LiDAR history matrix left and insert new column on right.
        Maintains temporal history for obstacle tracking.
        """
        self.lidar_matrix[:, :-1] = self.lidar_matrix[:, 1:]
        self.lidar_matrix[:, -1] = lidar_data
        return self.lidar_matrix

    def detect_obstacle(self, min_dist=0.0, max_dist=MIN_SAFE_DISTANCE, window=10, ratio=0.2):
        """
        Detect obstacles in LiDAR regions (left, center, right) using density ratio.
        Returns boolean flags and counts for each sub-region.
        """
        h, w = self.lidar_matrix.shape
        left_data = self.lidar_matrix[:45, -window:]
        center_data = self.lidar_matrix[45:135, -window:]
        right_data = self.lidar_matrix[135:, -window:]
        center_left = center_data[:45, :]
        center_right = center_data[45:, :]

        def region_has_obstacle(data):
            total = data.size
            close = np.sum((data > min_dist) & (data <= max_dist))
            return (close / total) >= ratio, close

        left_obs, left_count = region_has_obstacle(left_data)
        center_obs, center_count = region_has_obstacle(center_data)
        right_obs, right_count = region_has_obstacle(right_data)
        center_left_count = np.sum((center_left > min_dist) & (center_left <= max_dist))
        center_right_count = np.sum((center_right > min_dist) & (center_right <= max_dist))

        return (left_obs, center_obs, right_obs, left_count, center_left_count, center_right_count, right_count)

    def estimate_obstacle_size(self, min_distance=MIN_SAFE_DISTANCE, window=50):
        """
        Estimate obstacle width and persistence using LiDAR counts in regions.
        Returns estimated width (m), close timesteps, and primary region.
        """
        h, w = self.lidar_matrix.shape
        left_data = self.lidar_matrix[:45, -window:]
        center_data = self.lidar_matrix[45:135, -window:]
        right_data = self.lidar_matrix[135:, -window:]
        center_left = center_data[:45, :]
        center_right = center_data[45:, :]

        left_count = np.sum(left_data <= min_distance)
        center_left_count = np.sum(center_left <= min_distance)
        center_right_count = np.sum(center_right <= min_distance)
        right_count = np.sum(right_data <= min_distance)

        counts = {
            'left': left_count,
            'center_left': center_left_count,
            'center_right': center_right_count,
            'right': right_count
        }
        primary_region = max(counts, key=counts.get) if any(counts.values()) else 'center'

        if primary_region in ['left', 'center_left']:
            data = np.vstack((left_data, center_left)) if primary_region == 'center_left' else left_data
        elif primary_region in ['right', 'center_right']:
            data = np.vstack((right_data, center_right)) if primary_region == 'center_right' else right_data
        else:
            data = center_data

        close_beams = np.sum(np.any(data <= min_distance, axis=1))
        beam_angle = LIDAR_FOV / LIDAR_MATRIX_HEIGHT
        avg_distance = np.mean(data[data <= min_distance]) if np.any(data <= min_distance) else min_distance
        width = 2 * avg_distance * np.tan(close_beams * beam_angle / 2)
        close_timesteps = np.sum(np.any(self.lidar_matrix[np.arange(data.shape[0]), -window:] <= min_distance, axis=0))
        return width, close_timesteps, primary_region

    def _get_state(self):
        """
        Get current observation: resized/normalized camera image + LiDAR ranges.
        Handles missing devices gracefully.
        """
        image_data_cnn = np.zeros((64, 64, 3), dtype=np.float32)
        if self.camera:
            img_buf = self.camera.getImage()
            if img_buf:
                img = np.frombuffer(img_buf, dtype=np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                img_resized = cv2.resize(img_rgb, (64, 64), interpolation=cv2.INTER_AREA)
                image_data_cnn = img_resized.astype(np.float32) / 255.0

        lidar_data = np.full(LIDAR_MATRIX_HEIGHT, 100.0, dtype=np.float32)
        if self.lidar:
            raw_lidar = self.lidar.getRangeImage()
            if raw_lidar:
                lidar_data = np.array(raw_lidar, dtype=np.float32)
                lidar_data[np.isinf(lidar_data)] = 100.0


        return {"image": image_data_cnn, "lidar": lidar_data}

    def vision_based_action(self, state):
        """
        Compute expert action using vision (lane lines) and LiDAR (obstacles).
        Implements FSM logic for states: lane following, avoiding, straight, returning.
        Returns normalized [steering, speed] action.
        """
        lidar_data = state["lidar"]
        min_distance = np.min(lidar_data)
        self.lidar_matrix = self.update_lidar_matrix(lidar_data)

        steering = 0.0
        speed = MAX_SPEED / 2
        distance = 0.0

        # Get cropped bottom image for lane detection
        if self.camera:
            img_buf = self.camera.getImage()
            if img_buf is not None:
                img = np.frombuffer(img_buf, dtype=np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
                image_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                crop = image_rgb[IMAGE_HEIGHT-110:IMAGE_HEIGHT, :]
            else:
                crop = np.zeros((110, IMAGE_WIDTH, 3), dtype=np.uint8)
        else:
            crop = np.zeros((110, IMAGE_WIDTH, 3), dtype=np.uint8)

        # Obstacle detection and adaptation
        left_obs, center_obs, right_obs, left_count, center_left_count, center_right_count, right_count = \
            self.detect_obstacle(min_dist=0.0, max_dist=MIN_SAFE_DISTANCE)
        obstacle_detected = left_obs or center_obs or right_obs
        if obstacle_detected:
            obstacle_width, obstacle_length, primary_region = self.estimate_obstacle_size()
            self.dynamic_steering = min(MAX_STEERING_ANGLE, BASE_AVOIDANCE_STEERING * (1 + obstacle_width / 2))
            self.dynamic_straight_duration = max(BASE_STRAIGHT_DURATION, int(obstacle_length * (MAX_SPEED / speed)))
        else:
            self.dynamic_steering = BASE_AVOIDANCE_STEERING
            self.dynamic_straight_duration = BASE_STRAIGHT_DURATION

        if self.state == STATE_LANE_FOLLOWING:
            # Yellow lane override (e.g., for construction)
            yellow_mask = mask_yellow_lane(crop)
            yellow_indices = cv2.findNonZero(yellow_mask)
            yellow_override = False
            if yellow_indices is not None:
                yellow_center_x = int(np.mean(yellow_indices[:,0,0]))
                image_center_x = crop.shape[1] // 2
                if yellow_center_x > image_center_x:
                    steering = MAX_STEERING_ANGLE
                    yellow_override = True

            if not yellow_override:
                left_obs, center_obs, right_obs, left_count, center_left_count, center_right_count, right_count = \
                    self.detect_obstacle(min_dist=0.0, max_dist=MIN_SAFE_DISTANCE)
                obstacle_detected = left_obs or center_obs or right_obs
                if obstacle_detected:
                    speed = REDUCED_SPEED
                    pre = preprocess_image(crop)
                    if pre is not None:
                        lines = cv2.HoughLinesP(pre, 1, np.pi/60, 15,
                                                minLineLength=MIN_LINE_LENGTH,
                                                maxLineGap=MAX_LINE_GAP)
                        left_line, right_line = group_and_select_lines(lines, IMAGE_WIDTH)
                        distance = draw_and_smooth(crop.copy(), left_line, right_line,
                                                  IMAGE_WIDTH, 110,
                                                  alpha=SMOOTHING_ALPHA,
                                                  prev=self.prev_lines)
                        self.original_lane_center = distance
                    self.state = STATE_AVOIDING
                    self.avoidance_steps = 0
                    total_close = left_count + center_left_count + center_right_count + right_count
                    if right_count > 0.3 * total_close or (center_right_count > center_left_count and center_obs):
                        self.avoidance_side = "left"
                        steering = -self.dynamic_steering
                    elif left_count > 0.3 * total_close or (center_left_count > center_right_count and center_obs):
                        self.avoidance_side = "right"
                        steering = self.dynamic_steering
                    else:
                        if center_left_count < center_right_count:
                            self.avoidance_side = "left"
                            steering = -self.dynamic_steering
                        else:
                            self.avoidance_side = "right"
                            steering = self.dynamic_steering
                else:
                    # Warning handling
                    left_warn, center_warn, right_warn, left_warn_count, center_left_warn_count, center_right_warn_count, right_warn_count = \
                        self.detect_obstacle(min_dist=MIN_SAFE_DISTANCE, max_dist=WARNING_DISTANCE)
                    warn_detected = left_warn or center_warn or right_warn
                    if warn_detected:
                        left_score = left_warn_count + center_left_warn_count
                        right_score = right_warn_count + center_right_warn_count
                        if left_score > right_score:
                            target_distance = -50
                        elif right_score > left_score:
                            target_distance = 50
                        else:
                            target_distance = 0
                    else:
                        target_distance = 0
                    speed = MAX_SPEED / 2
                    pre = preprocess_image(crop)
                    if pre is not None:
                        lines = cv2.HoughLinesP(pre, 1, np.pi/60, 15,
                                                minLineLength=MIN_LINE_LENGTH,
                                                maxLineGap=MAX_LINE_GAP)
                        left_line, right_line = group_and_select_lines(lines, IMAGE_WIDTH)
                        distance = draw_and_smooth(crop.copy(), left_line, right_line,
                                                  IMAGE_WIDTH, 110,
                                                  alpha=SMOOTHING_ALPHA,
                                                  prev=self.prev_lines)
                        steering = np.clip(((distance - target_distance) / (IMAGE_WIDTH/3)) * MAX_STEERING_ANGLE,
                                           -MAX_STEERING_ANGLE,
                                           MAX_STEERING_ANGLE)
                    else:
                        steering = 0.0

        elif self.state == STATE_AVOIDING:
            self.avoidance_steps += 1
            speed = REDUCED_SPEED
            left_obs2, _, right_obs2, left_count2, _, _, right_count2 = self.detect_obstacle(
                min_dist=0.0, max_dist=2.0
            )
            if self.pending_side_switch_steps > 0:
                self.pending_side_switch_steps -= 1
                steering = 0.0
                if self.pending_side_switch_steps == 0 and self.pending_switch_to is not None:
                    self.avoidance_side = self.pending_switch_to
                    steering = self.dynamic_steering * SMOOTH_AVOIDANCE_FACTOR \
                        if self.avoidance_side == "right" else -self.dynamic_steering * SMOOTH_AVOIDANCE_FACTOR
                    self.avoidance_steps = 0
                    self.pending_switch_to = None
            else:
                if self.avoidance_side == "left" and left_obs2 and self.pending_side_switch_steps == 0:
                    self.pending_side_switch_steps = 50
                    self.pending_switch_to = "right"
                elif self.avoidance_side == "right" and right_obs2 and self.pending_side_switch_steps == 0:
                    self.pending_side_switch_steps = 50
                    self.pending_switch_to = "left"
                steering = -self.dynamic_steering if self.avoidance_side == "left" else self.dynamic_steering
            # Transition to straight if clear
            if self.avoidance_steps >= RETURN_THRESHOLD and not any(
                self.detect_obstacle(min_dist=0.0, max_dist=MIN_SAFE_DISTANCE)[:3]):
                self.state = STATE_DRIVING_STRAIGHT
                self.avoidance_steps = 0
                steering = 0.0

        elif self.state == STATE_DRIVING_STRAIGHT:
            self.avoidance_steps += 1
            speed = REDUCED_SPEED
            steering = 0.0
            if self.avoidance_steps >= self.dynamic_straight_duration:
                self.state = STATE_RETURNING
                self.avoidance_steps = 0
                steering = -self.dynamic_steering if self.avoidance_side == "left" else self.dynamic_steering

        elif self.state == STATE_RETURNING:
            self.avoidance_steps += 1
            speed = REDUCED_SPEED
            left_obs, center_obs, right_obs, left_count, center_left_count, center_right_count, right_count = \
                self.detect_obstacle(min_dist=0.0, max_dist=MIN_SAFE_DISTANCE)
            obstacle_detected = left_obs or center_obs or right_obs
            if obstacle_detected:
                total_close = left_count + center_left_count + center_right_count + right_count
                if right_count > 0.3 * total_close or (center_right_count > center_left_count and center_obs):
                    self.avoidance_side = "left"
                    steering = -self.dynamic_steering
                elif left_count > 0.3 * total_close or (center_left_count > center_right_count and center_obs):
                    self.avoidance_side = "right"
                    steering = self.dynamic_steering
                else:
                    if center_left_count < center_right_count:
                        self.avoidance_side = "left"
                        steering = -self.dynamic_steering
                    else:
                        self.avoidance_side = "right"
                        steering = self.dynamic_steering
                self.state = STATE_AVOIDING
                self.avoidance_steps = 0
            else:
                pre = preprocess_image(crop)
                if pre is not None:
                    lines = cv2.HoughLinesP(pre, 1, np.pi/60, 15,
                                            minLineLength=MIN_LINE_LENGTH,
                                            maxLineGap=MAX_LINE_GAP)
                    left_line, right_line = group_and_select_lines(lines, IMAGE_WIDTH)
                    distance = draw_and_smooth(crop.copy(), left_line, right_line,
                                              IMAGE_WIDTH, 110,
                                              alpha=SMOOTHING_ALPHA,
                                              prev=self.prev_lines)
                    target_distance = self.original_lane_center if self.original_lane_center is not None else 0
                    steering = np.clip(((distance - target_distance) / (IMAGE_WIDTH/3)) * MAX_STEERING_ANGLE,
                                       -MAX_STEERING_ANGLE,
                                       MAX_STEERING_ANGLE)
                    if abs(distance - target_distance) < 20 and self.avoidance_steps >= RETURN_THRESHOLD:
                        self.state = STATE_LANE_FOLLOWING
                        self.avoidance_side = None
                        self.original_lane_center = None
                        speed = MAX_SPEED / 2
                else:
                    steering = -self.dynamic_steering if self.avoidance_side == "left" else self.dynamic_steering

        # Normalize actions
        steering_normalized = np.clip(steering / MAX_STEERING_ANGLE, -1.0, 1.0)
        speed_normalized = (speed / MAX_SPEED) * 2 - 1
        self.last_distance_error = distance

        return np.array([steering_normalized, speed_normalized], dtype=np.float32)

    def calculate_reward(self, distance_error, min_lidar_distance, collision, current_state):
        """
        Compute dense environment reward based on state, safety, and progress.
        State-specific bonuses; penalties for proximity/collision.
        Ensures minimum positive reward in safe conditions.
        """
        reward = 0
        center_distance = abs(distance_error)
        
        # Safety reward tiering
        if min_lidar_distance > 4.0:
            safety_reward = 10
        elif min_lidar_distance > 3.0:
            safety_reward = 8
        elif min_lidar_distance > 2.0:
            safety_reward = 5
        elif min_lidar_distance > 1.5:
            safety_reward = 2
        else:
            safety_reward = 0
        reward += safety_reward
        
        # State-specific rewards
        if current_state == STATE_LANE_FOLLOWING:
            if center_distance < 10:
                reward += 20
            elif center_distance < 20:
                reward += 15
            elif center_distance < 30:
                reward += 8
            else:
                reward += 3
            if center_distance < 15:
                reward += 5
                
        elif current_state == STATE_AVOIDING:
            base_avoiding_reward = 15
            if min_lidar_distance > 3.0:
                base_avoiding_reward += 10
            elif min_lidar_distance > 2.0:
                base_avoiding_reward += 5
            reward += base_avoiding_reward
            if center_distance < 60:
                reward += 5
                
        elif current_state == STATE_DRIVING_STRAIGHT:
            base_straight_reward = 12
            if center_distance < 40:
                base_straight_reward += 8
            elif center_distance < 60:
                base_straight_reward += 4
            reward += base_straight_reward
            
        elif current_state == STATE_RETURNING:
            if center_distance < 10:
                proximity_reward = 30
            elif center_distance < 20:
                proximity_reward = 25
            elif center_distance < 35:
                proximity_reward = 20
            elif center_distance < 50:
                proximity_reward = 15
            elif center_distance < 80:
                proximity_reward = 10
            else:
                proximity_reward = 5
            reward += proximity_reward
            if hasattr(self, 'last_center_distance_returning'):
                if center_distance < self.last_center_distance_returning:
                    reward += 15
                elif center_distance == self.last_center_distance_returning:
                    reward += 3
            self.last_center_distance_returning = center_distance
            if center_distance < 15:
                reward += 25
            if min_lidar_distance > 3.0:
                reward += 12
            elif min_lidar_distance > 2.0:
                reward += 8
            else:
                reward += 4
            reward += 8
        
        # Penalties
        if collision:
            reward -= 100
        elif min_lidar_distance < 1.0:
            reward -= 20
        elif min_lidar_distance < 1.5:
            reward -= 5
        
        # Minimum reward floor
        if not collision and min_lidar_distance >= 1.0:
            if current_state == STATE_RETURNING:
                reward = max(10, reward)
            else:
                reward = max(3, reward)
        
        reward = np.clip(reward, -100, 100)
        return reward

    def safe_features_extraction(self, features_extractor, observations):
        """
        Safely extract features from observations, handling tensor shape conversions.
        Raises errors on invalid shapes for debugging.
        """
        try:
            img = observations["image"]
            if img.dim() == 4 and img.shape[1] != 3 and img.shape[3] == 3:
                img = img.permute(0, 3, 1, 2)
            elif img.dim() == 3 and img.shape[2] == 3:
                img = img.permute(2, 0, 1).unsqueeze(0)
            if img.shape[1] != 3:
                raise ValueError(f"Expected 3 channels, got {img.shape[1]} channels. Shape: {img.shape}")
            safe_observations = {"image": img, "lidar": observations["lidar"]}
            return features_extractor(safe_observations)
        except Exception as e:
            print(f"[SAFE-FE-ERROR] Error in safe_features_extraction: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def monitor_state_rewards(self, reward, distance_error, min_lidar_distance):
        """
        Monitor and log rewards per state for debugging (called externally if needed).
        Warns on negative rewards in returning state.
        """
        if not hasattr(self, 'step_count'):
            self.step_count = 0
        self.step_count += 1
        
        if self.step_count % 100 == 0:
            print(f"[REWARD-MONITOR] Step {self.step_count} | State: {self.state} | "
                  f"Reward: {reward:.2f} | Distance: {abs(distance_error):.1f} | "
                  f"LiDAR: {min_lidar_distance:.2f}")
        
        if self.state == STATE_RETURNING and reward < 0:
            print(f"🚨 CRITICAL: Negative reward {reward:.2f} in STATE_RETURNING at step {self.step_count}!")
            print(f"    Distance: {abs(distance_error):.1f}, Min LiDAR: {min_lidar_distance:.2f}")

    def render(self, mode='human'):
        """
        Render the environment: display cropped image with lane visualization.
        Supports 'human' (imshow) or 'rgb_array' (return image).
        """
        if self.camera is None:
            print("[RENDER] No camera available for rendering.")
            return None

        img_buf = self.camera.getImage()
        if img_buf is None:
            print("[RENDER] Failed to get camera image for rendering.")
            return None

        try:
            img = np.frombuffer(img_buf, dtype=np.uint8).reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            crop = img_rgb[IMAGE_HEIGHT-110:IMAGE_HEIGHT, :]
            
            # Apply lane detection visualization
            pre = preprocess_image(crop)
            if pre is not None:
                lines = cv2.HoughLinesP(pre, 1, np.pi/60, 15,
                                        minLineLength=MIN_LINE_LENGTH,
                                        maxLineGap=MAX_LINE_GAP)
                left_line, right_line = group_and_select_lines(lines, IMAGE_WIDTH)
                draw_and_smooth(crop.copy(), left_line, right_line,
                               IMAGE_WIDTH, 110,
                               alpha=SMOOTHING_ALPHA,
                               prev=self.prev_lines)
            
            if mode == 'human':
                cv2.imshow('Autonomous Driving Environment', crop)
                cv2.waitKey(1)
            elif mode == 'rgb_array':
                return crop
            else:
                raise ValueError(f"Unsupported render mode: {mode}")
        
        except Exception as e:
            print(f"[RENDER] Error during rendering: {e}")
            import traceback
            traceback.print_exc()
            return None

    def close(self):
        """
        Clean up: disable devices, close windows.
        """
        try:
            if hasattr(self, 'camera') and self.camera is not None:
                self.camera.disable()
            if hasattr(self, 'lidar') and self.lidar is not None:
                self.lidar.disable()
            if hasattr(cv2, 'destroyAllWindows'):
                cv2.destroyAllWindows()
            print("[CLOSE] Environment closed successfully.")
        except Exception as e:
            print(f"[CLOSE] Error during environment cleanup: {e}")
            import traceback
            traceback.print_exc()

    def save_buffers(self, path_prefix="buffer"):
        """
        Save all state buffers to pickle files.
        """
        try:
            for state_type, buffer in self.state_buffers.items():
                if len(buffer) > 0:
                    file_path = f"{path_prefix}_{state_type.replace(' ', '_').lower()}_buffer.pkl"
                    with open(file_path, 'wb') as f:
                        pickle.dump(list(buffer), f)
                    print(f"[SAVE] Saved {state_type} buffer to {file_path} | Size: {len(buffer)}")
        except Exception as e:
            print(f"[SAVE] Error saving buffers: {e}")
            import traceback
            traceback.print_exc()

    def load_buffers(self, path_prefix="buffer"):
        """
        Load state buffers from pickle files, respecting maxlen.
        """
        try:
            for state_type in self.state_buffers.keys():
                file_path = f"{path_prefix}_{state_type.replace(' ', '_').lower()}_buffer.pkl"
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        loaded_buffer = pickle.load(f)
                        self.state_buffers[state_type].clear()
                        self.state_buffers[state_type].extend(loaded_buffer[:self.state_buffers[state_type].maxlen])
                    print(f"[LOAD] Loaded {state_type} buffer from {file_path} | Size: {len(self.state_buffers[state_type])}")
        except Exception as e:
            print(f"[LOAD] Error loading buffers: {e}")
            import traceback
            traceback.print_exc()

    def get_buffer_stats(self):
        """
        Compute and return statistics for each state buffer (size, avg priority, etc.).
        """
        stats = {}
        for state_type, buffer in self.state_buffers.items():
            stats[state_type] = {
                'size': len(buffer),
                'avg_priority': np.mean([exp['priority'] for exp in buffer]) if buffer else 0.0,
                'avg_action_diversity': np.mean([exp['action_diversity'] for exp in buffer]) if buffer else 0.0,
                'imitation_count': len([exp for exp in buffer if exp['phase'] == 'imitation']),
                'mixed_count': len([exp for exp in buffer if exp['phase'] == 'mixed'])
            }
        return stats

    def log_performance_metrics(self):
        """
        Log key metrics: avg rewards, BC loss, buffer stats, recent performance.
        """
        try:
            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
                print(f"[METRICS] Average episode reward (last 10): {avg_reward:.2f}")
            
            if len(self.bc_losses) > 0:
                avg_bc_loss = np.mean(self.bc_losses[-20:]) if len(self.bc_losses) >= 20 else np.mean(self.bc_losses)
                print(f"[METRICS] Average BC loss (last 20): {avg_bc_loss:.6f}")
            
            buffer_stats = self.get_buffer_stats()
            print(f"[METRICS] Buffer stats: {buffer_stats}")
            
            if self.recent_rewards:
                recent_avg = np.mean(list(self.recent_rewards)[-10:])
                print(f"[METRICS] Recent rewards (last 10): {recent_avg:.2f}")
        
        except Exception as e:
            print(f"[METRICS] Error logging performance metrics: {e}")
            import traceback
            traceback.print_exc()

    def check_environment_health(self):
        """
        Check and log status of key components (devices, model, buffers).
        """
        health_status = {
            'camera': self.camera is not None and self.camera.isEnabled(),
            'lidar': self.lidar is not None and self.lidar.isEnabled(),
            'car_node': self.car_node is not None,
            'model': self.model is not None,
            'discriminator': self.discriminator is not None,
            'disc_features_extractor': self.disc_features_extractor is not None,
            'buffer_sizes': {k: len(v) for k, v in self.state_buffers.items()}
        }
        print(f"[HEALTH] Environment health check: {health_status}")
        return health_status


class CustomCNNWithLiDAR(BaseFeaturesExtractor):
    """
    Custom feature extractor for SB3: CNN for image + MLP for LiDAR, then combined MLP.
    Handles shape conversions in forward pass.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        super(CustomCNNWithLiDAR, self).__init__(observation_space, features_dim)
        self.image_shape = observation_space.spaces["image"].shape
        self.lidar_shape = observation_space.spaces["lidar"].shape
        
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            pytorch_image_shape = (self.image_shape[2], self.image_shape[0], self.image_shape[1])
            sample_input = torch.zeros(1, *pytorch_image_shape)
            cnn_output_dim = self.cnn(sample_input).shape[1]
        
        self.lidar_mlp = nn.Sequential(
            nn.Linear(self.lidar_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        combined_dim = cnn_output_dim + 32
        self.combined_mlp = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations):
        img = observations["image"]
        lidar = observations["lidar"]
        # Handle batch and channel dimensions
        if img.dim() == 4:
            if img.shape[1] != 3:
                if img.shape[3] == 3:
                    img = img.permute(0, 3, 1, 2)
        elif img.dim() == 3:
            if img.shape[2] == 3:
                img = img.permute(2, 0, 1).unsqueeze(0)
            else:
                img = img.unsqueeze(0)
        if img.shape[1] != 3:
            raise ValueError(f"Expected 3 channels, got {img.shape[1]} channels. Shape: {img.shape}")
        
        img_features = self.cnn(img)
        if lidar.dim() == 1:
            lidar = lidar.unsqueeze(0)
        lidar_features = self.lidar_mlp(lidar)
        img_flat = torch.flatten(img_features, start_dim=1)
        combined = torch.cat([img_flat, lidar_features], dim=1)
        final_features = self.combined_mlp(combined)
        return final_features

def moving_average(data, window=50):
    """
    Compute simple moving average for smoothing plots.
    Handles short data gracefully.
    """
    if len(data) == 0:
        return []
    if len(data) < window:
        return list(np.convolve(data, np.ones(len(data)) / len(data), mode='valid'))
    return list(np.convolve(data, np.ones(window) / window, mode='valid'))

class ImitationLearningCallback(BaseCallback):
    """
    Custom SB3 callback for imitation learning: logs metrics, triggers BC, plots progress.
    Tracks rewards, losses, safety, states; generates comprehensive plots every interval.
    """
    def __init__(self, env, model, save_path="training_logs", 
                 bc_interval=200, plot_interval=2000, verbose=1):
        super().__init__(verbose)
        self.plot_interval = plot_interval  # حالا هر 200 مرحله رسم می‌شود
        self.env = env
        self.model = model
        self.save_path = os.path.abspath(save_path)
        self.bc_interval = bc_interval
        self.plot_interval = plot_interval
        self.episode_rewards = []
        self.bc_losses = []
        self.phase_transitions = []
        self.action_similarities = []
        self.safety_scores = []
        # Additional tracking for evaluation parameters
        self.step_rewards = []
        self.env_step_rewards = []
        self.irl_step_rewards = []
        self.min_lidar_distances = []
        self.distance_errors = []
        self.collisions = 0
        self.collision_timesteps = []
        self.state_occurrences = collections.Counter()
        self.buffer_size_history = []
        self.phase_stats = {
            'imitation': {'episodes': 0, 'avg_reward': 0, 'safety_incidents': 0},
            'mixed': {'episodes': 0, 'avg_reward': 0, 'safety_incidents': 0},
        }
        self.total_timesteps = 0
        self.current_episode_reward = 0.0
        self.previous_multiples = {"imitation": 0, "mixed": 0}
        self.bc_intervals = {"imitation": 1000, "mixed": 500}
        os.makedirs(self.save_path, exist_ok=True)
        if hasattr(env, 'set_model'):
            env.set_model(model)
        
    def _on_step(self):
        """
        Per-step callback: accumulate rewards, track metrics, handle episode ends, plot periodically.
        """
        self.total_timesteps += 1
        info = self.locals.get('infos', [{}])[0]
        reward = self.locals.get('rewards', [0])[0]
        self.current_episode_reward += reward

        # Debug: Log info and reward every 100 steps
        if self.total_timesteps % 100 == 0:
            print(f"[DEBUG-STEP] Timestep {self.total_timesteps} | Reward: {reward:.2f} | Info keys: {list(info.keys())}")

        # Collect metrics continuously with debug
        if 'reward_info' in info:
            reward_info = info['reward_info']
            similarity = reward_info.get('similarity', None)
            safety = reward_info.get('safety', None)
            self.action_similarities.append(similarity if similarity is not None else 0.0)  # Explicit default
            self.safety_scores.append(safety if safety is not None else 0.0)  # Explicit default
            self.step_rewards.append(reward_info.get('total', 0))
            self.env_step_rewards.append(reward_info.get('state_reward', 0))
            self.irl_step_rewards.append(reward_info.get('irl_reward', 0))
            self.min_lidar_distances.append(info.get('min_distance', 100))
            self.distance_errors.append(abs(info.get('lane_distance', 0)))
            if info.get('collision', False):
                self.collisions += 1
                self.collision_timesteps.append(self.total_timesteps)
            current_state = info.get('current_state', 'Unknown')
            self.state_occurrences[current_state] += 1
            print(f"[DEBUG-REWARD_INFO] Timestep {self.total_timesteps} | reward_info: {reward_info} | similarity: {similarity}, safety: {safety}")
        else:
            print(f"[WARNING] No reward_info in info at timestep {self.total_timesteps}")

        # Episode end handling
        if self.locals.get('dones', [False])[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.env.episode_rewards.append(self.current_episode_reward)
            self.env.recent_rewards.append(self.current_episode_reward)
            current_phase = self.env.current_phase
            self.phase_stats[current_phase]['episodes'] += 1
            self.phase_stats[current_phase]['avg_reward'] = (
                (self.phase_stats[current_phase]['avg_reward'] * 
                (self.phase_stats[current_phase]['episodes'] - 1) + 
                self.current_episode_reward) / 
                self.phase_stats[current_phase]['episodes']
            )
            if 'min_distance' in info and info['min_distance'] < 2.0:
                self.phase_stats[current_phase]['safety_incidents'] += 1
            print(f"[DEBUG-EPISODE] Episode {len(self.episode_rewards)} ended | Reward: {self.current_episode_reward:.2f}")
            self.current_episode_reward = 0.0
            if len(self.episode_rewards) % 10 == 0:
                recent_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
                print(f"\n[PROGRESS] Episode {len(self.episode_rewards)} | "
                    f"Phase: {current_phase} | "
                    f"Recent avg reward: {recent_reward:.3f} | "
                    f"Buffer sizes: { {k: len(v) for k, v in self.env.state_buffers.items()} }")

        # Trigger plotting
        if self.total_timesteps % self.plot_interval == 0:
            self._plot_training_progress()

        # Trigger BC training if needed
        if self.env.should_run_bc_training():
            self.env.train_behavioral_cloning_structured()

        return True
    
    def _on_rollout_end(self):
        """
        End-of-rollout: trigger BC if interval crossed and buffer sufficient.
        """
        phase = self.env.current_phase
        interval = self.bc_intervals[phase]
        current_multiples = self.num_timesteps // interval
        previous = self.previous_multiples[phase]
        if current_multiples > previous and sum(len(buf) for buf in self.env.state_buffers.values()) >= 64:
            print(f"[BC] Starting training at timestep {self.num_timesteps} (crossed {current_multiples * interval})")
            self.env.train_behavioral_cloning_structured()
            self.previous_multiples[phase] = current_multiples

    def _plot_training_progress(self):
        """
        Generate and save a 3x3 subplot dashboard of training metrics using Matplotlib.
        Plots ALL data from start to current timestep for continuous updates.
        """
        # Track buffer history
        current_buffer_sizes = {k: len(v) for k, v in self.env.state_buffers.items()}
        self.buffer_size_history.append(current_buffer_sizes)

        # Debug: Log data sizes for verification
        print(f"[PLOT-DEBUG] Timestep {self.total_timesteps} | "
            f"Episode Rewards: {len(self.episode_rewards)} | "
            f"BC Losses: {len(self.env.bc_losses)} | "
            f"Action Similarities: {len(self.action_similarities)} | "
            f"Safety Scores: {len(self.safety_scores)} | "
            f"Step Rewards: {len(self.step_rewards)}")

        # Create figure
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle(f'Imitation Learning Progress - Timestep {self.total_timesteps} (All Data from Start)', 
                    fontsize=16, fontweight='bold')

        # 1. Episode Rewards (All episodes from start)
        if self.episode_rewards:
            axes[0, 0].plot(self.episode_rewards, alpha=0.6, linewidth=1, color='blue', label='Raw')
            if len(self.episode_rewards) >= 20:
                ma = moving_average(self.episode_rewards, 20)
                axes[0, 0].plot(range(19, len(self.episode_rewards)), ma, 'r-', linewidth=2, label='MA (20)')
            axes[0, 0].set_title('Episode Rewards (All from Start)')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No Episode Rewards Data', ha='center', va='center')

        # 2. Behavioral Cloning Loss (All iterations from start)
        if self.env.bc_losses:
            axes[0, 1].plot(self.env.bc_losses, 'g-', alpha=0.7, label='Raw')
            if len(self.env.bc_losses) >= 10:
                ma_loss = moving_average(self.env.bc_losses, 10)
                axes[0, 1].plot(range(9, len(self.env.bc_losses)), ma_loss, 'darkgreen', linewidth=2, label='MA (10)')
            axes[0, 1].set_title('Behavioral Cloning Loss (All from Start)')
            axes[0, 1].set_xlabel('BC Iteration')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_yscale('log')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No BC Loss Data', ha='center', va='center')

        # 3. Action Similarity to Expert (All timesteps from start)
        if self.action_similarities:
            axes[0, 2].plot(self.action_similarities, alpha=0.6, color='purple', label='Raw')
            if len(self.action_similarities) >= 50:
                ma_sim = moving_average(self.action_similarities, 50)
                axes[0, 2].plot(range(len(self.action_similarities) - len(ma_sim), len(self.action_similarities)), ma_sim, 'b-', linewidth=2, label='MA (50)')
            axes[0, 2].set_title('Action Similarity to Expert (All from Start)')
            axes[0, 2].set_xlabel('Timestep')
            axes[0, 2].set_ylabel('Similarity')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].text(0.5, 0.5, 'No Action Similarity Data', ha='center', va='center')

        # 4. Safety Scores (All steps from start)
        if self.safety_scores:
            axes[1, 0].plot(self.safety_scores, alpha=0.6, color='orange', label='Raw')
            if len(self.safety_scores) >= 50:
                ma_safety = moving_average(self.safety_scores, 50)
                axes[1, 0].plot(range(len(self.safety_scores) - len(ma_safety), len(self.safety_scores)), ma_safety, 'red', linewidth=2, label='MA (50)')
            axes[1, 0].set_title('Safety Scores (All from Start)')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Safety Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Safety Scores Data', ha='center', va='center')

        # 5. Reward Components (All steps from start)
        if self.step_rewards:
            axes[1, 1].plot(self.step_rewards, label='Total Reward', alpha=0.7, color='blue')
            axes[1, 1].plot(self.env_step_rewards, label='Env Reward', alpha=0.7, color='green')
            axes[1, 1].plot(self.irl_step_rewards, label='IRL Reward', alpha=0.7, color='red')
            axes[1, 1].set_title('Reward Components (All from Start)')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Reward')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Reward Components Data', ha='center', va='center')

        # 6. Min LiDAR Distances (All steps from start)
        if self.min_lidar_distances:
            axes[1, 2].plot(self.min_lidar_distances, alpha=0.6, color='purple', label='Raw')
            if len(self.min_lidar_distances) >= 50:
                ma_lidar = moving_average(self.min_lidar_distances, 50)
                axes[1, 2].plot(range(len(self.min_lidar_distances) - len(ma_lidar), len(self.min_lidar_distances)), ma_lidar, 'darkviolet', linewidth=2, label='MA (50)')
            axes[1, 2].set_title('Min LiDAR Distances (All from Start)')
            axes[1, 2].set_xlabel('Step')
            axes[1, 2].set_ylabel('Distance')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'No LiDAR Distance Data', ha='center', va='center')

        # 7. Distance Errors (All steps from start)
        if self.distance_errors:
            axes[2, 0].plot(self.distance_errors, alpha=0.6, color='brown', label='Raw')
            if len(self.distance_errors) >= 50:
                ma_dist = moving_average(self.distance_errors, 50)
                axes[2, 0].plot(range(len(self.distance_errors) - len(ma_dist), len(self.distance_errors)), ma_dist, 'saddlebrown', linewidth=2, label='MA (50)')
            axes[2, 0].set_title('Lane Distance Errors (All from Start)')
            axes[2, 0].set_xlabel('Step')
            axes[2, 0].set_ylabel('Error')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        else:
            axes[2, 0].text(0.5, 0.5, 'No Distance Error Data', ha='center', va='center')

        # 8. Average Reward by Phase (Cumulative)
        phases = list(self.phase_stats.keys())
        avg_rewards = [self.phase_stats[p]['avg_reward'] for p in phases]
        axes[2, 1].bar(phases, avg_rewards, alpha=0.7, color=['blue', 'green'])
        axes[2, 1].set_title('Average Reward by Phase (Cumulative)')
        axes[2, 1].set_ylabel('Average Reward')
        axes[2, 1].grid(True, alpha=0.3)

        # 9. State Distribution Pie Chart (Cumulative occurrences)
        if self.state_occurrences:
            state_names = list(self.state_occurrences.keys())
            state_counts = list(self.state_occurrences.values())
            axes[2, 2].pie(state_counts, labels=state_names, autopct='%1.1f%%', startangle=90)
            axes[2, 2].set_title('State Distribution (All Occurrences from Start)')
        else:
            axes[2, 2].text(0.5, 0.5, 'No State Distribution Data', ha='center', va='center')

        # Collision rate annotation
        collision_rate = self.collisions / max(self.total_timesteps, 1) * 1000  # per 1000 steps
        axes[2, 1].text(0.5, 0.5, f'Collision Rate: {collision_rate:.2f}/1000 steps', 
                        ha='center', va='center', transform=axes[2, 1].transAxes, fontsize=12, color='red')

        # Save and optional live display
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(self.save_path, f'training_progress_t{self.total_timesteps}.png')
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[PLOT] Saved continuous training progress plot (all data) to {save_path}")
            # Optional: Live display for debugging (uncomment if needed)
            # plt.show(block=False)
            # plt.pause(0.1)
        except Exception as e:
            print(f"[PLOT-ERROR] Failed to save plot: {e}")
        plt.close(fig)

def main():
    """
    Main training script: initialize env/model, load checkpoints, train with callback, save.
    Handles resume from saved model and buffers.
    """
    print("="*60)
    print("🚗 AUTONOMOUS DRIVING - RESUME IMITATION LEARNING VIA RL")
    print("="*60)
    
    env = AutonomousDrivingEnv()
    env.seed(SEED)
    print(f"[INIT] Environment created")
    
    # Load model or initialize new
    try:
        model = PPO.load("./final_imitation_model", env=env)
        print(f"[INIT] Loaded saved model successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}. Initializing new model.")
        import traceback
        traceback.print_exc()
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs={
                "features_extractor_class": CustomCNNWithLiDAR,
                "features_extractor_kwargs": {"features_dim": 512},
                "net_arch": [dict(pi=[256, 128], vf=[256, 128])]
            },
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            seed=SEED,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    env.set_model(model)
    env.expert_policy = env.create_expert_policy()
    print(f"[INIT] Model configured in environment")
    
    # Load experience buffers
    try:
        with open("experience_buffer.pkl", "rb") as f:
            loaded_buffers = pickle.load(f)
            for state, buffer_data in loaded_buffers.items():
                if state in env.state_buffers:
                    # Convert loaded list to deque
                    env.state_buffers[state] = collections.deque(buffer_data, maxlen=env.state_buffers[state].maxlen)
        print(f"[INIT] Loaded experience buffer successfully")
        print(f"[INIT] Buffer sizes: { {k: len(v) for k, v in env.state_buffers.items()} }")
    except Exception as e:
        print(f"[WARNING] Failed to load experience buffer: {e}. Starting with empty buffers.")
        import traceback
        traceback.print_exc()
    
    # Initialize callback
    callback = ImitationLearningCallback(
        env=env,
        model=model,
        save_path="./imitation_learning_logs",
        bc_interval=200,
        plot_interval=10000,
        verbose=1
    )
    
    total_timesteps = 60000
    print(f"[TRAINING] Starting/resuming training for {total_timesteps} timesteps")
    print(f"[TRAINING] Phase schedule:")
    print(f"  - Imitation: 0 -> {env.phase_config['imitation_duration']}")
    print(f"  - Mixed: {env.phase_config['imitation_duration']} -> {total_timesteps}")
    
    # Train
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=False  # Resume without resetting timestep counter
        )
        print("[SUCCESS] Training completed!")
        model.save("./final_imitation_model")
        with open("experience_buffer.pkl", "wb") as f:
            pickle.dump({k: list(v) for k, v in env.state_buffers.items()}, f)
        print("[SAVE] Model and experience buffer saved successfully")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()

    print(f"\n🏁 Training {'completed successfully' if success else 'failed'}")