"""
PPO Agent Implementation for Microgrid Energy Management
Proximal Policy Optimization - ON-POLICY algorithm
Key differences from SAC/TD3:
- Uses rollout buffer instead of replay buffer
- Clipped surrogate objective
- Multiple epochs on same data
- GAE for advantage estimation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class RolloutBuffer:
    """
    Buffer for on-policy PPO algorithm
    Stores complete trajectories for advantage estimation
    """
    def __init__(self, buffer_size, state_dim, action_dim, gamma=0.99, gae_lambda=0.95):
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.trajectory_start_idx = 0
        
    def add(self, state, action, reward, value, log_prob, done):
        """Add transition to buffer with overflow protection"""
        if self.ptr >= self.buffer_size:
            return False  # Buffer full
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr += 1
        return True
    
    def compute_returns_and_advantages(self, last_value):
        """
        Compute GAE (Generalized Advantage Estimation) advantages and returns
        This is done after collecting a full trajectory
        """
        # Convert last_value to numpy if it's a tensor
        if torch.is_tensor(last_value):
            last_value = last_value.cpu().numpy()
        
        last_gae_lam = 0
        for step in reversed(range(self.ptr)):
            if step == self.ptr - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]
            
            # TD error: delta = r + gamma * V(s') - V(s)
            delta = self.rewards[step] + self.gamma * next_value * next_non_terminal - self.values[step]
            
            # GAE: A = delta + gamma * lambda * A(t+1)
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        # Returns = advantages + values (only for valid indices)
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]
    
    def get(self, batch_size):
        """
        Generator that yields random mini-batches
        """
        indices = np.arange(self.ptr)
        np.random.shuffle(indices)
        
        for start_idx in range(0, self.ptr, batch_size):
            end_idx = min(start_idx + batch_size, self.ptr)
            batch_indices = indices[start_idx:end_idx]
            
            yield (
                torch.FloatTensor(self.states[batch_indices]),
                torch.FloatTensor(self.actions[batch_indices]),
                torch.FloatTensor(self.log_probs[batch_indices]),
                torch.FloatTensor(self.advantages[batch_indices]),
                torch.FloatTensor(self.returns[batch_indices])
            )
    
    def reset(self):
        """Clear buffer after updates"""
        self.ptr = 0
        self.trajectory_start_idx = 0
    
    def is_full(self):
        """Check if buffer is full"""
        return self.ptr >= self.buffer_size
    
    def size(self):
        """Return current buffer size"""
        return self.ptr


class ActorCritic(nn.Module):
    """
    PPO Actor-Critic Network with SEPARATE networks for actor and critic
    - Actor: Stochastic policy (Gaussian) - 3 layer MLP
    - Critic: State value function V(s) - 3 layer MLP
    Separate networks often work better for PPO!
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # SEPARATE Actor Network (3 layers for better capacity)
        self.actor_fc1 = nn.Linear(state_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.policy_mean = nn.Linear(hidden_dim // 2, action_dim)
        # State-dependent log_std for adaptive exploration
        self.policy_log_std = nn.Linear(hidden_dim // 2, action_dim)
        
        # SEPARATE Critic Network (3 layers)
        self.critic_fc1 = nn.Linear(state_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Orthogonal initialization for better training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # Smaller initialization for output heads
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.policy_log_std.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
    
    def forward(self, state):
        """Forward pass through both actor and critic (SEPARATE paths)"""
        # Actor path
        a = F.relu(self.actor_fc1(state))
        a = F.relu(self.actor_fc2(a))
        a = F.relu(self.actor_fc3(a))
        action_mean = self.policy_mean(a)
        # State-dependent std with WIDER bounds for exploration
        action_log_std = torch.clamp(self.policy_log_std(a), -2, 1.0)  # Wider bounds!
        action_std = torch.exp(action_log_std)
        # Minimum std floor to ensure exploration never dies
        action_std = torch.clamp(action_std, min=0.1)
        
        # Critic path (completely separate)
        c = F.relu(self.critic_fc1(state))
        c = F.relu(self.critic_fc2(c))
        c = F.relu(self.critic_fc3(c))
        value = self.value_head(c)
        
        return action_mean, action_std, value
    
    def get_action(self, state, deterministic=False):
        """
        Sample action from policy
        Returns: action, log_prob, value
        """
        action_mean, action_std, value = self.forward(state)
        
        if deterministic:
            # For evaluation: use mean action
            action = torch.tanh(action_mean)
            return action, None, value
        
        # Sample from Gaussian distribution
        dist = Normal(action_mean, action_std)
        action_sample = dist.rsample()  # Reparameterization trick
        
        # Apply tanh squashing
        action = torch.tanh(action_sample)
        
        # Compute log probability with tanh correction
        log_prob = dist.log_prob(action_sample).sum(dim=-1, keepdim=True)
        
        # Tanh correction for log_prob
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value
    
    def evaluate_actions(self, states, actions):
        """
        Evaluate actions for PPO update
        Returns: log_probs, values, entropy
        """
        action_mean, action_std, values = self.forward(states)
        
        dist = Normal(action_mean, action_std)
        
        # Inverse tanh to get pre-tanh actions
        atanh_actions = torch.atanh(torch.clamp(actions, -0.999, 0.999))
        
        # Log probability
        log_probs = dist.log_prob(atanh_actions).sum(dim=-1, keepdim=True)
        
        # Tanh correction
        log_probs -= torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        
        # Entropy for exploration bonus
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, values, entropy


class PPOAgent:
    """
    PPO Agent with clipped surrogate objective
    Now with SEPARATE optimizers for actor and critic!
    """
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        
        # Actor-Critic network
        self.policy = ActorCritic(
            config.STATE_DIM,
            config.ACTION_DIM,
            config.HIDDEN_DIM
        ).to(device)
        
        # SEPARATE optimizers for actor and critic (better training!)
        actor_params = list(self.policy.actor_fc1.parameters()) + \
                       list(self.policy.actor_fc2.parameters()) + \
                       list(self.policy.actor_fc3.parameters()) + \
                       list(self.policy.policy_mean.parameters()) + \
                       list(self.policy.policy_log_std.parameters())
        
        critic_params = list(self.policy.critic_fc1.parameters()) + \
                        list(self.policy.critic_fc2.parameters()) + \
                        list(self.policy.critic_fc3.parameters()) + \
                        list(self.policy.value_head.parameters())
        
        self.actor_optimizer = optim.Adam(actor_params, lr=config.ACTOR_LR, eps=1e-5)
        self.critic_optimizer = optim.Adam(critic_params, lr=config.CRITIC_LR, eps=1e-5)
        
        # Learning rate scheduler (linear decay)
        self.lr_scheduler = None  # Can be added for long training
        
        # KL divergence target - DISABLE early stopping to allow full updates
        self.target_kl = None  # Disabled - let PPO learn aggressively
        
        # Rollout buffer
        self.buffer = RolloutBuffer(
            config.N_STEPS,
            config.STATE_DIM,
            config.ACTION_DIM,
            config.GAMMA,
            config.GAE_LAMBDA
        )
        
        # PPO hyperparameters
        self.clip_epsilon = config.CLIP_EPSILON
        self.value_coef = config.VALUE_COEF
        self.entropy_coef = config.ENTROPY_COEF
        self.max_grad_norm = config.MAX_GRAD_NORM
        self.n_epochs = config.N_EPOCHS
        self.batch_size = config.BATCH_SIZE
        
        # Tracking
        self.total_steps = 0
    
    def select_action(self, state, deterministic=False):
        """Select action from policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.policy.get_action(state_tensor, deterministic)
            
            action = action.cpu().numpy()[0]
            value = value.cpu().item() if value is not None else 0
            log_prob = log_prob.cpu().item() if log_prob is not None else 0
            
        return action * self.config.ACTION_SCALE, value, log_prob
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition in rollout buffer"""
        # Undo action scaling for storage
        action = action / self.config.ACTION_SCALE
        self.buffer.add(state, action, reward, value, log_prob, done)
        self.total_steps += 1
    
    def update(self, next_state):
        """
        PPO update after collecting N_STEPS of data
        """
        # Get value of last state for GAE
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            _, _, last_value = self.policy.get_action(next_state_tensor, deterministic=True)
            last_value = last_value.cpu().item()
        
        # Compute advantages and returns
        self.buffer.compute_returns_and_advantages(last_value)
        
        # Multiple epochs of updates
        policy_losses = []
        value_losses = []
        entropies = []
        
        kl_exceeded = False
        for epoch in range(self.n_epochs):
            if kl_exceeded:
                break  # Early stopping if KL divergence too high
            
            # Mini-batch updates
            for batch in self.buffer.get(self.batch_size):
                states, actions, old_log_probs, advantages, returns = batch
                
                states = states.to(self.device)
                actions = actions.to(self.device)
                old_log_probs = old_log_probs.to(self.device)
                advantages = advantages.to(self.device)
                returns = returns.to(self.device)
                
                # Normalize advantages (important for stable training)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Evaluate actions with current policy
                log_probs, values, entropy = self.policy.evaluate_actions(states, actions)
                
                # PPO clipped surrogate objective
                log_ratio = log_probs - old_log_probs.unsqueeze(1)
                ratio = torch.exp(log_ratio)
                
                # KL early stopping (disabled for aggressive learning)
                if self.target_kl is not None:
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    if approx_kl > 1.5 * self.target_kl:
                        kl_exceeded = True
                        break
                
                surr1 = ratio * advantages.unsqueeze(1)
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.unsqueeze(1)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Clipped value loss (helps with value function stability)
                value_loss = F.mse_loss(values, returns.unsqueeze(1))
                
                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()
                
                # SEPARATE optimizer updates (better training!)
                # Update actor
                actor_loss = policy_loss + self.entropy_coef * entropy_loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Update critic
                critic_loss = self.value_coef * value_loss
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(-entropy_loss.item())
        
        # Clear buffer
        self.buffer.reset()
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies)
        }
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        # Handle both old and new checkpoint formats
        if 'actor_optimizer_state_dict' in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        elif 'optimizer_state_dict' in checkpoint:
            pass  # Old format, skip optimizer loading
        self.total_steps = checkpoint.get('total_steps', 0)
        print(f"Model loaded from {filepath}")
    
    def is_ready_to_update(self):
        """Check if buffer has enough data for update"""
        return self.buffer.size() >= self.config.N_STEPS
