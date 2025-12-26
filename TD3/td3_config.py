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

    ACTOR_LR = 3e-4           # Balanced learning
    CRITIC_LR = 3e-4          # Balanced learning

    GAMMA = 0.99              # Standard discount
    TAU = 0.005               # Standard soft update
    POLICY_NOISE = 0.2        # Balanced exploration
    NOISE_CLIP = 0.5          # Standard clipping
    POLICY_FREQ = 2
    ACTION_NOISE = 0.1        # Balanced exploration

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
