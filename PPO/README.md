# PPO (Proximal Policy Optimization) for Microgrid Energy Management

## Algorithm Overview

**PPO** is an **on-policy** reinforcement learning algorithm that uses a clipped surrogate objective to ensure stable policy updates. Unlike SAC and TD3 (which are off-policy), PPO collects fresh trajectories for each update.

## Key Differences from SAC and TD3

### Architecture Comparison

| Feature | SAC | TD3 | PPO |
|---------|-----|-----|-----|
| **Policy Type** | Stochastic (Gaussian) | Deterministic | Stochastic (Gaussian) |
| **Learning** | Off-policy | Off-policy | **On-policy** |
| **Storage** | Replay Buffer | Replay Buffer | **Rollout Buffer** |
| **Critics** | 2 Q-networks | 2 Q-networks | **1 Value network V(s)** |
| **Update** | Every step | Delayed | **Multiple epochs** |
| **Objective** | Entropy-regularized | TD error | **Clipped surrogate** |

### PPO-Specific Features

1. **Clipped Objective**: Prevents large policy updates
   ```
   L_CLIP = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
   ```

2. **GAE (Generalized Advantage Estimation)**: Better advantage calculation
   ```
   A_t = Σ (γλ)^l * δ_{t+l}
   where δ_t = r_t + γV(s_{t+1}) - V(s_t)
   ```

3. **On-Policy Learning**: Uses fresh trajectories, discards old data after updates

4. **Multiple Epochs**: Reuses collected data for several mini-batch updates

## Algorithm Flow

```
1. Collect N_STEPS of experience:
   - For each step:
     * Sample action from policy: a ~ π(·|s)
     * Execute action, observe reward and next state
     * Store (s, a, r, V(s), log π(a|s))

2. Compute advantages using GAE:
   - Calculate TD errors: δ_t = r_t + γV(s_{t+1}) - V(s_t)
   - Compute GAE: A_t = Σ (γλ)^l * δ_{t+l}
   - Compute returns: R_t = A_t + V(s_t)

3. Update policy for N_EPOCHS:
   - For each mini-batch:
     * Compute new log probabilities and values
     * Calculate importance ratio: ratio = π_new / π_old
     * Compute clipped policy loss
     * Compute value loss
     * Add entropy bonus
     * Backpropagate and update

4. Discard old data, repeat from step 1
```

## Hyperparameters

### PPO-Specific
- `CLIP_EPSILON = 0.2`: Clipping range for policy updates
- `GAE_LAMBDA = 0.95`: GAE lambda for advantage smoothing
- `N_STEPS = 2048`: Trajectory length before update
- `N_EPOCHS = 10`: Number of update epochs per rollout
- `ENTROPY_COEF = 0.01`: Entropy regularization coefficient

### Standard RL
- `GAMMA = 0.99`: Discount factor
- `ACTOR_LR = 3e-4`: Policy network learning rate
- `CRITIC_LR = 3e-4`: Value network learning rate
- `BATCH_SIZE = 64`: Mini-batch size for updates

## Advantages of PPO

1. **Stable Training**: Clipping prevents destructive policy updates
2. **Sample Efficient**: Multiple epochs on same data
3. **Simple Implementation**: Single network with shared features
4. **No Replay Buffer**: Lower memory footprint
5. **Robust**: Works well across many environments

## Disadvantages

1. **On-Policy**: Requires fresh samples (can't reuse old data indefinitely)
2. **Slower per Episode**: Collects full trajectories before update
3. **Hyperparameter Sensitive**: Clip epsilon and GAE lambda need tuning

## File Structure

```
PPO/
├── __init__.py              # Package initialization
├── ppo_config.py            # PPO hyperparameters
├── ppo_agent.py             # PPO algorithm implementation
│   ├── RolloutBuffer        # On-policy trajectory storage
│   ├── ActorCritic          # Combined policy and value network
│   └── PPOAgent             # Main training logic
├── ppo_main.py              # Training script
└── README.md                # This file
```

## Usage

### Training
```bash
python PPO/ppo_main.py
```

### Evaluation
```python
from PPO.ppo_agent import PPOAgent
from PPO.ppo_config import PPOConfig

config = PPOConfig()
agent = PPOAgent(config)
agent.load('./models/ppo_microgrid/ppo_final.pt')

# Evaluate
state = env.reset()
action, _, _ = agent.select_action(state, deterministic=True)
```

## Implementation Details

### Network Architecture
```
Input (31-dim state)
    ↓
Shared FC1 (256) + Tanh
    ↓
Shared FC2 (256) + Tanh
    ↓
    ├─→ Policy Mean (6-dim)
    │   + Learnable Log Std
    │   → Gaussian Distribution
    │
    └─→ Value FC (256) + Tanh
        → Value Head (1-dim)
```

### GAE Computation
```python
δ_t = r_t + γV(s_{t+1}) - V(s_t)
A_t = δ_t + γλ * A_{t+1}
```

### Clipped Loss
```python
ratio = exp(log π_new(a|s) - log π_old(a|s))
L = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
```

## Expected Results

Based on realistic hyperparameters:
- **Cost**: ~$88,000 - $90,000 per day
- **Convergence**: ~200-300 episodes
- **Stability**: Lower variance than SAC due to clipping
- **Performance**: Comparable to SAC, better than TD3 on this task

## References

1. Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
2. Schulman et al. (2016) - "High-Dimensional Continuous Control Using Generalized Advantage Estimation"

## Comparison with SAC/TD3

**When to use PPO:**
- Need stable, predictable training
- Limited replay memory
- Prefer simplicity over complexity
- Environment has low-dimensional actions

**When to use SAC:**
- Need maximum sample efficiency
- Can afford replay buffer
- Continuous control with entropy regularization

**When to use TD3:**
- Deterministic policies preferred
- Stability more important than exploration
- Overestimation is a concern
