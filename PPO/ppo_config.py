"""
PPO Configuration for Microgrid Energy Management
PPO is an ON-POLICY algorithm with different design than SAC/TD3
"""

import sys
import os

# Import base config from SAC for environment settings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SAC'))
from sac_config import SACConfig

class PPOConfig(SACConfig):
    """PPO-specific hyperparameters"""
    
    # Environment settings (inherited from SACConfig)
    # STATE_DIM = 31
    # ACTION_DIM = 6
    
    # PPO-Specific Parameters
    ALGORITHM = "PPO"
    
    # Network Architecture
    HIDDEN_DIM = 256        # Standard size, stability over capacity
    
    # Learning Rates - Start higher, will decay
    ACTOR_LR = 3e-4          # Standard actor LR
    CRITIC_LR = 1e-3         # Higher critic LR
    
    # PPO Clipping - slightly looser for more aggressive updates
    CLIP_EPSILON = 0.2       # Standard PPO clipping
    CLIP_VALUE = True        # Whether to clip value function loss
    VALUE_CLIP = 0.2         # Standard value clip
    
    # GAE (Generalized Advantage Estimation)
    GAMMA = 0.99            # Standard discount
    GAE_LAMBDA = 0.95       # Standard GAE lambda
    
    # Training Parameters - KEY: More data per update!
    BATCH_SIZE = 64         # Larger batches for stability
    N_STEPS = 480           # 20 episodes per update - more data!
    N_EPOCHS = 10           # Standard epochs
    MAX_GRAD_NORM = 0.5     # Gradient clipping
    
    # Entropy Regularization - CRITICAL for exploration
    ENTROPY_COEF = 0.02     # Moderate entropy
    VALUE_COEF = 0.5        # Standard value coefficient
    
    # ACTION SCALE - THIS IS KEY! PPO needs bigger action range
    ACTION_SCALE = 0.15     # 15% capacity per step (was 6%)
    
    # Episodes and Evaluation
    TOTAL_EPISODES = 500
    EVAL_FREQUENCY = 25     # Evaluate every N episodes
    EVAL_EPISODES = 3       # Number of episodes for evaluation
    
    # Action scaling
    ACTION_SCALE = 0.06     # Scale actions to valid range
    
    # Model Saving
    SAVE_FREQUENCY = 100    # Save model every N episodes
    
    # Allow override via environment variable
    def __init__(self):
        super().__init__()
        import os
        if 'PPO_EPISODES' in os.environ:
            self.TOTAL_EPISODES = int(os.environ['PPO_EPISODES'])
        # PPO doesn't use replay buffer (on-policy)
        self.BUFFER_SIZE = None
