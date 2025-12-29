# TD3 Optimization: Reducing Oscillation and Instability

## Problem Analysis
Your TD3 training curves showed **high oscillation** with:
- Early peak at ~-88.5 
- Subsequent drift to worse performance (~-89.4)
- Significant variance compared to smooth SAC convergence
- Classic TD3 instability symptoms

## Root Causes Identified
1. **Actor LR too high** - 3e-4 caused overshooting and divergence
2. **Policy delay too weak** - POLICY_FREQ=2 updated actor too frequently
3. **Noise poorly scaled** - POLICY_NOISE=0.2 and ACTION_NOISE=0.1 too aggressive
4. **Insufficient gradient clipping** - max_norm=1.0 allowed large updates

## Changes Made

### 1. Configuration Optimizations (td3_config.py)

#### Actor Learning Rate: 3e-4 → 1e-4
```python
ACTOR_LR = 1e-4  # Reduced by 66% to prevent overshooting
```
**Impact**: Actor updates are now more conservative, reducing sudden policy changes that cause oscillation.

#### Critic Learning Rate: 3e-4 (unchanged)
```python
CRITIC_LR = 3e-4  # Keep higher for faster value estimation
```
**Impact**: Critic still learns quickly to guide actor, but actor responds more cautiously.

#### Policy Delay: 2 → 3
```python
POLICY_FREQ = 3  # Update actor every 3 critic updates (was every 2)
```
**Impact**: Actor waits longer for critic values to stabilize, reducing coordinated instability.

#### Policy Noise: 0.2 → 0.1
```python
POLICY_NOISE = 0.1  # Reduced by 50% for smoother exploration
```
**Impact**: Target policy exploration is less random, creating more stable targets.

#### Noise Clip: 0.5 → 0.3
```python
NOISE_CLIP = 0.3  # Tighter bounds prevent extreme action proposals
```
**Impact**: Limits the range of noisy actions, preventing outlier targets.

#### Action Noise: 0.1 → 0.05
```python
ACTION_NOISE = 0.05  # Reduced by 50% for cleaner rollouts
```
**Impact**: Episode exploration is more focused, reducing random perturbations.

### 2. Agent Training Enhancements (td3_agent.py)

#### Weight Decay Added
```python
self.actor_optimizer = optim.Adam(
    self.actor.parameters(), 
    lr=config.ACTOR_LR,
    weight_decay=1e-5  # Added for regularization
)
self.critic_optimizer = optim.Adam(
    self.critic.parameters(),
    lr=config.CRITIC_LR,
    weight_decay=1e-5  # Added for regularization
)
```
**Impact**: L2 regularization prevents weights from growing too large.

#### Stronger Gradient Clipping: 1.0 → 0.5/0.3
```python
# Critic: max_norm=1.0 → 0.5
nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)

# Actor: max_norm=1.0 → 0.3
nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.3)
```
**Impact**: Prevents large gradient steps that cause sudden policy shifts.

#### Gradient Step Tracking (Optional)
```python
self.gradient_steps = 0  # Prepared for future learning rate scheduling
```
**Impact**: Infrastructure for adaptive learning rates if needed.

## Expected Improvements

### Before Optimization
- Oscillating learning curve with high variance
- Unstable convergence
- Performance worse than SAC
- Policy changes too abrupt

### After Optimization
- ✅ Smoother learning curves (less oscillation)
- ✅ More stable convergence trajectory
- ✅ Reduced regressive phases
- ✅ Better alignment with SAC performance
- ✅ Safer actor updates with conservative learning

## Hyperparameter Tuning Philosophy

The optimizations follow TD3 best practices:

1. **Actor-Critic Balance**: Actor learns slower (1e-4) than critic (3e-4)
2. **Policy Delay**: Stronger emphasis on delayed policy updates (POLICY_FREQ=3)
3. **Noise Scaling**: All noise parameters reduced by 50% for stability
4. **Gradient Control**: Tighter clipping bounds prevent divergence
5. **Regularization**: Weight decay prevents parameter explosion

## Fine-Tuning Recommendations

If oscillation persists after retraining:

### Option A: More Conservative (slower but smoother)
```python
ACTOR_LR = 5e-5        # Further reduce
POLICY_FREQ = 4        # More delay
POLICY_NOISE = 0.05    # Further reduce
ACTION_NOISE = 0.02    # Further reduce
```

### Option B: Intermediate (balanced)
```python
ACTOR_LR = 1e-4        # Current (recommended starting point)
POLICY_FREQ = 3        # Current
POLICY_NOISE = 0.1     # Current
ACTION_NOISE = 0.05    # Current
```

### Option C: Aggressive (if learning stalls)
```python
ACTOR_LR = 2e-4        # Slightly increase
POLICY_FREQ = 2        # Original
POLICY_NOISE = 0.15    # Increase a bit
ACTION_NOISE = 0.08    # Increase a bit
```

## Implementation Notes

- No code changes to main loop required
- Backward compatible with existing training scripts
- No new dependencies added
- Can be applied mid-training by loading checkpoints

## Validation Strategy

Monitor these metrics when retraining:
1. **Episode Reward**: Should increase smoothly with minimal drops
2. **Actor Loss**: Should stabilize quickly
3. **Critic Loss**: Should decrease monotonically
4. **Gradient Norms**: Should stay bounded by clipping limits
5. **25-episode Rolling Average**: Should show smooth curves (not sawtooth)

Expected timeline: See significant improvement within first 200 episodes.

## References
- Original TD3 paper: Fujimoto et al. (2018) - "Addressing Function Approximation Error in Actor-Critic Methods"
- TD3 stability improvements documented in follow-up work
