# SAC Configuration for Multi-Microgrid Energy Management
# Replace your Q_Learning_v7.py with this SAC implementation

import os
import torch

class SACConfig:
    """Configuration for SAC algorithm and environment"""
    
    # Environment Parameters
    # Input data file with load, PV, and price information (use file located in this package)
    DATA_PATH = os.path.join(os.path.dirname(__file__), 'Data_for_Qcode.csv')
    ENERGY_DATA_PATH = r'Energy_data_v7.csv'  # Output file for energy calculations
    
    # Microgrid Constants (from your code_v7.py)
    ESS_MAX = 100.0  # kWh
    EV_MAX = 16.0    # kWh
    MBP_MAX = 8.5    # Maximum market buying price
    MBP_MIN = 4.0    # Minimum market buying price
    CHP_MIN = 80.0   # Minimum CHP generation
    CHP_MAX = 200.0  # Maximum CHP generation
    N_C = 0.9        # Charging efficiency
    N_D = 0.9        # Discharging efficiency
    
    # Number of microgrids
    NUM_INDUSTRY = 2    # ind1, ind2
    NUM_COMMUNITY = 2   # com3, com4
    NUM_SINGLE_D = 3    # sd5, sd6, sd7
    NUM_CAMPUS = 1      # camp8
    TOTAL_MG = 8
    
    # State space dimension
    # For each MG: ESS/EV level (6), loads (8), PV (8), prices (4), time (1), stress (1)
    STATE_DIM = (
        6 +  # ESS levels: com1, com2, camp; EV levels: sd1, sd2, sd3
        8 +  # Load for all 8 MGs
        8 +  # PV generation for all 8 MGs
        4 +  # gbp, mbp, msp, gsp
        2 +  # time_of_day, day_of_week
        2 +  # total_deficit, total_surplus
        1    # stress = total_surplus / total_deficit
    )  # Total: 31 dimensions
    
    # Action space dimension
    # Continuous actions for: bidding strategy coefficients (alpha1, alpha2) for each MG
    # OR continuous ESS/EV charging rates
    # Option 1: Bidding actions (2 per MG = 16 actions)
    # Option 2: ESS/EV charging actions (6 actions)
    # We'll use Option 2 for direct energy management
    ACTION_DIM = 6  # [com1_rate, com2_rate, sd1_rate, sd2_rate, sd3_rate, camp_rate]
    
    # SAC Hyperparameters
    GAMMA = 0.99              # Discount factor (higher for long-term planning)
    TAU = 0.005               # Target network soft update rate
    ACTOR_LR = 3e-4           # Actor learning rate (balanced)
    CRITIC_LR = 3e-4          # Critic learning rate (balanced)
    ALPHA_LR = 3e-4           # Entropy temperature learning rate
    INITIAL_ALPHA = 0.2       # Initial entropy temperature
    AUTO_ENTROPY = True       # Automatic entropy tuning
    TARGET_ENTROPY = -ACTION_DIM  # Target entropy = -action_dim
    
    # Action scaling (realistic for stable optimization)
    ACTION_SCALE = 0.06       # Charge/discharge rate as fraction of capacity per step
    
    # Reward shaping weights
    USE_REWARD_SHAPING = True  # Enable improved reward function
    REWARD_SCALE = 1000.0      # Scale down costs for better learning
    
    # Network Architecture
    HIDDEN_DIM = 256          # Hidden layer size
    NUM_HIDDEN_LAYERS = 2     # Number of hidden layers
    ACTIVATION = 'relu'       # Activation function
    
    # Training Parameters
    BATCH_SIZE = 256          # Batch size for training (balanced)
    BUFFER_SIZE = 300_000     # Replay buffer size (increased but realistic)
    WARMUP_STEPS = 1000       # Random exploration steps before training
    TRAIN_FREQ = 1            # Update frequency (steps)
    GRADIENT_STEPS = 1        # Number of gradient steps per update
    
    # Training Schedule
    # Allow overriding via environment variable `SAC_EPISODES` for quick runs
    try:
        TOTAL_EPISODES = int(os.environ.get("SAC_EPISODES", 500))
    except Exception:
        TOTAL_EPISODES = 500     # More episodes for better convergence
    MAX_STEPS_PER_EPISODE = 24  # 24 hourly time slots per day
    EVAL_FREQ = 25            # Evaluate every 25 episodes
    SAVE_FREQ = 100           # Model save frequency (episodes)
    
    # Logging
    LOG_DIR = './logs/sac_microgrid'
    MODEL_DIR = './models/sac_microgrid'
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Reward Shaping
    CONSTRAINT_PENALTY = -1000.0  # Penalty for violating ESS/EV limits
    UNMET_DEMAND_PENALTY = -100.0  # Penalty per kWh of unmet demand
    
    @staticmethod
    def normalize_state(state_dict):
        """Normalize state values to [-1, 1] or [0, 1]"""
        normalized = []
        
        # Normalize ESS/EV levels to [0, 1]
        normalized.append(state_dict['com1_ess'] / SACConfig.ESS_MAX)
        normalized.append(state_dict['com2_ess'] / SACConfig.ESS_MAX)
        normalized.append(state_dict['camp_ess'] / SACConfig.ESS_MAX)
        normalized.append(state_dict['sd1_ev'] / SACConfig.EV_MAX)
        normalized.append(state_dict['sd2_ev'] / SACConfig.EV_MAX)
        normalized.append(state_dict['sd3_ev'] / SACConfig.EV_MAX)
        
        # Add loads (normalized by max expected load, e.g., 500 kWh)
        MAX_LOAD = 500.0
        for i in range(1, 9):
            load_key = f'load_mg{i}'
            normalized.append(min(state_dict.get(load_key, 0) / MAX_LOAD, 1.0))
        
        # Add PV generation (normalized by max expected PV, e.g., 300 kWh)
        MAX_PV = 300.0
        for i in range(1, 9):
            pv_key = f'pv_mg{i}'
            normalized.append(min(state_dict.get(pv_key, 0) / MAX_PV, 1.0))
        
        # Normalize prices to [0, 1]
        normalized.append((state_dict['gbp'] - 4.0) / 8.0)  # Assuming price range 4-12
        normalized.append((state_dict['mbp'] - SACConfig.MBP_MIN) / (SACConfig.MBP_MAX - SACConfig.MBP_MIN))
        normalized.append((state_dict['msp'] - 2.0) / 8.0)
        normalized.append((state_dict['gsp'] - 4.0) / 8.0)
        
        # Normalize time features
        normalized.append(state_dict['time_of_day'] / 24.0)  # [0, 1]
        normalized.append(state_dict['day_of_week'] / 7.0)   # [0, 1]
        
        # Normalize energy deficit/surplus (assume max 2000 kWh)
        MAX_ENERGY = 2000.0
        normalized.append(min(state_dict['total_deficit'] / MAX_ENERGY, 1.0))
        normalized.append(min(state_dict['total_surplus'] / MAX_ENERGY, 1.0))
        
        # Stress (already a ratio)
        normalized.append(min(state_dict['stress'], 1.0))
        
        return normalized
    
    @staticmethod
    def denormalize_action(action):
        """Convert action from [-1, 1] to actual charging/discharging rate"""
        # action in [-1, 1]: -1 = full discharge, 0 = idle, +1 = full charge
        return action  # Keep in [-1, 1] for now, will be scaled in environment
