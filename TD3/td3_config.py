import os
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SAC.sac_config import SACConfig


class TD3Config(SACConfig):
    """Configuration for TD3 algorithm on the microgrid environment."""

    # Reduced actor LR to prevent oscillation - key fix for stability
    ACTOR_LR = 1e-4           # Reduced from 3e-4 to prevent overshooting
    CRITIC_LR = 3e-4          # Keep critic LR higher for faster value learning

    GAMMA = 0.99              # Standard discount
    TAU = 0.005               # Standard soft update (conservative)
    
    # Noise parameters - carefully tuned to reduce oscillation
    POLICY_NOISE = 0.1        # Reduced from 0.2 - smoother exploration
    NOISE_CLIP = 0.3          # Reduced from 0.5 - limit extreme actions
    POLICY_FREQ = 3           # Increased from 2 - delay actor updates more
    ACTION_NOISE = 0.05       # Reduced from 0.1 - less exploration noise during rollout

    try:
        TOTAL_EPISODES = int(os.environ.get("TD3_EPISODES", 500))
    except Exception:
        TOTAL_EPISODES = 500     # Extended training
    MAX_STEPS_PER_EPISODE = 24
    EVAL_FREQ = 25
    SAVE_FREQ = 100

    BATCH_SIZE = 256
    BUFFER_SIZE = 200_000
    WARMUP_STEPS = 1_000
    GRADIENT_STEPS = 1

    LOG_DIR = "./logs/td3_microgrid"
    MODEL_DIR = "./models/td3_microgrid"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
