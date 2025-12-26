"""
PPO Training Script for Microgrid Energy Management
On-policy training with trajectory collection and multi-epoch updates
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'SAC'))

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from PPO.ppo_config import PPOConfig
from PPO.ppo_agent import PPOAgent
from SAC.microgrid_env import MicrogridEnv


def evaluate_policy(agent, env, num_episodes=3):
    """
    Evaluate policy performance (deterministic actions)
    """
    total_rewards = []
    total_costs = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _, _ = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        total_costs.append(env.episode_cost)
    
    return np.mean(total_rewards), np.mean(total_costs)


def train_ppo():
    """Main PPO training loop"""
    config = PPOConfig()
    
    # Create environment
    env = MicrogridEnv(config.DATA_PATH, config)
    
    # Create agent
    agent = PPOAgent(config, device='cpu')
    
    # Create directories
    os.makedirs('./models/ppo_microgrid', exist_ok=True)
    os.makedirs('./logs/ppo_microgrid', exist_ok=True)
    
    # Tracking
    episode_rewards = []
    episode_costs = []
    eval_rewards = []
    eval_costs = []
    policy_losses = []
    value_losses = []
    
    print("=" * 60)
    print("Starting PPO Training for Multi-Microgrid Energy Management")
    print("=" * 60)
    print(f"Algorithm: {config.ALGORITHM} (On-Policy)")
    print(f"State Dimension: {config.STATE_DIM}")
    print(f"Action Dimension: {config.ACTION_DIM}")
    print(f"Total Episodes: {config.TOTAL_EPISODES}")
    print(f"Device: cpu")
    print(f"Action Scale: {config.ACTION_SCALE*100:.1f}% of capacity per step")
    print("-" * 60)
    print(f"Rollout Steps: {config.N_STEPS}")
    print(f"Update Epochs: {config.N_EPOCHS}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Clip Epsilon: {config.CLIP_EPSILON}")
    print(f"GAE Lambda: {config.GAE_LAMBDA}")
    print(f"Entropy Coef: {config.ENTROPY_COEF}")
    print(f"Actor LR: {config.ACTOR_LR}")
    print(f"Critic LR: {config.CRITIC_LR}")
    print("=" * 60)
    
    episode = 0
    total_timesteps = 0
    
    while episode < config.TOTAL_EPISODES:
        # Reset environment
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        # Run episode
        while not done:
            # Select action
            action, value, log_prob = agent.select_action(state, deterministic=False)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            state = next_state
            episode_reward += reward
            steps += 1
            total_timesteps += 1
            
            # Update policy when buffer has enough data
            if agent.buffer.size() >= config.N_STEPS:
                update_info = agent.update(next_state)
                policy_losses.append(update_info['policy_loss'])
                value_losses.append(update_info['value_loss'])
        
        # Track episode metrics
        episode_rewards.append(episode_reward)
        episode_costs.append(env.episode_cost)
        episode += 1
        
        # Print progress - adaptive frequency based on total episodes
        log_freq = 100 if config.TOTAL_EPISODES >= 5000 else 10
        if episode % log_freq == 0:
            avg_reward = np.mean(episode_rewards[-log_freq:])
            avg_cost = np.mean(episode_costs[-log_freq:])
            
            # Calculate % profit vs baseline (no control = $90,692)
            baseline_cost = 90692.0
            profit_pct = ((baseline_cost - avg_cost) / baseline_cost) * 100
            
            # Get latest losses if available
            latest_policy_loss = policy_losses[-1] if policy_losses else 0
            latest_value_loss = value_losses[-1] if value_losses else 0
            
            print(f"Episode {episode}/{config.TOTAL_EPISODES} | "
                  f"Steps: {steps} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg Reward ({log_freq}): {avg_reward:.2f} | "
                  f"Cost: {avg_cost:.2f} | "
                  f"Profit: {profit_pct:+.2f}% | "
                  f"P_Loss: {latest_policy_loss:.4f} | "
                  f"V_Loss: {latest_value_loss:.4f}")
        
        # Evaluation - adaptive frequency for long runs
        eval_freq = 500 if config.TOTAL_EPISODES >= 5000 else config.EVAL_FREQUENCY
        if episode % eval_freq == 0:
            print("\n" + "=" * 60)
            print(f"Evaluation after Episode {episode}")
            eval_reward, eval_cost = evaluate_policy(agent, env, config.EVAL_EPISODES)
            eval_rewards.append(eval_reward)
            eval_costs.append(eval_cost)
            baseline_cost = 90692.0
            eval_profit_pct = ((baseline_cost - eval_cost) / baseline_cost) * 100
            print(f"Avg Evaluation Reward: {eval_reward:.2f}")
            print(f"Avg Evaluation Cost: {eval_cost:.2f}")
            print(f"Avg Profit: {eval_profit_pct:+.2f}%")
            print("=" * 60 + "\n")
        
        # Save model - adaptive frequency for long runs
        save_freq = 1000 if config.TOTAL_EPISODES >= 5000 else config.SAVE_FREQUENCY
        if episode % save_freq == 0:
            agent.save(f'./models/ppo_microgrid/ppo_episode_{episode}.pt')
    
    # Save final model
    agent.save('./models/ppo_microgrid/ppo_final.pt')
    
    # Plot training curves
    plot_training_curves(
        episode_rewards,
        episode_costs,
        eval_rewards,
        eval_costs,
        policy_losses,
        value_losses,
        config
    )
    
    print("\nTraining completed successfully!")


def plot_training_curves(episode_rewards, episode_costs, eval_rewards, eval_costs,
                         policy_losses, value_losses, config):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.6, label='Episode Reward')
    if len(episode_rewards) > 10:
        smoothed = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
        axes[0, 0].plot(range(9, len(episode_rewards)), smoothed, label='Smoothed', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Episode costs
    axes[0, 1].plot(episode_costs, alpha=0.6, label='Episode Cost')
    if len(episode_costs) > 10:
        smoothed = np.convolve(episode_costs, np.ones(10)/10, mode='valid')
        axes[0, 1].plot(range(9, len(episode_costs)), smoothed, label='Smoothed', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Cost ($)')
    axes[0, 1].set_title('Episode Costs')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Evaluation metrics
    eval_episodes = [i * config.EVAL_FREQUENCY for i in range(1, len(eval_rewards) + 1)]
    axes[0, 2].plot(eval_episodes, eval_rewards, marker='o', label='Eval Reward')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Reward')
    axes[0, 2].set_title('Evaluation Rewards')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Evaluation costs
    axes[1, 0].plot(eval_episodes, eval_costs, marker='o', color='red', label='Eval Cost')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Cost ($)')
    axes[1, 0].set_title('Evaluation Costs')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Policy loss
    if policy_losses:
        axes[1, 1].plot(policy_losses, alpha=0.7, label='Policy Loss')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Policy Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    # Value loss
    if value_losses:
        axes[1, 2].plot(value_losses, alpha=0.7, color='green', label='Value Loss')
        axes[1, 2].set_xlabel('Update')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].set_title('Value Loss')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'./logs/ppo_microgrid/training_curves_ppo_{timestamp}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    train_ppo()
