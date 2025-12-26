# SAC Training Script for Multi-Microgrid Energy Management
# This replaces your main_v7.py

import numpy as np
import torch
import os
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import SAC components
from sac_config import SACConfig
from sac_agent import SACAgent
from microgrid_env import MicrogridEnv

def train_sac():
    """Main training loop for SAC agent"""
    
    # Initialize configuration
    config = SACConfig()
    
    # Create directories
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # Initialize environment
    env = MicrogridEnv(config.DATA_PATH, config)
    
    # Initialize SAC agent
    agent = SACAgent(config)
    
    # Training statistics
    episode_rewards = []
    episode_costs = []
    actor_losses = []
    critic_losses = []
    
    print("="*60)
    print("Starting SAC Training for Multi-Microgrid Energy Management")
    print("="*60)
    print(f"State Dimension: {config.STATE_DIM}")
    print(f"Action Dimension: {config.ACTION_DIM}")
    print(f"Total Episodes: {config.TOTAL_EPISODES}")
    print(f"Device: {config.DEVICE}")
    print(f"Action Scale: {config.ACTION_SCALE * 100:.1f}% of capacity per step")
    print(f"Reward Shaping: {'Enabled' if config.USE_REWARD_SHAPING else 'Disabled'}")
    print("="*60)
    
    # Show baseline targets
    try:
        import pandas as pd
        baseline_df = pd.read_csv('./logs/baseline_comparison.csv')
        best_baseline = baseline_df['Cost'].min()
        print(f"\n📊 Performance Targets:")
        print(f"   Best Baseline: ${best_baseline:.2f}")
        print(f"   Target (10% improvement): ${best_baseline * 0.9:.2f}")
        print("="*60 + "\n")
    except:
        pass
    
    # Training loop
    for episode in range(config.TOTAL_EPISODES):
        # Reset environment
        state, info = env.reset()
        episode_reward = 0
        episode_cost = 0
        episode_steps = 0
        
        # Episode loop (24 hours)
        for step in range(config.MAX_STEPS_PER_EPISODE):
            # Select action
            if agent.total_steps < config.WARMUP_STEPS:
                # Random exploration during warmup
                action = env.action_space.sample()
            else:
                # Use policy
                action = agent.select_action(state, deterministic=False)
            
            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store transition in replay buffer
            done = terminated or truncated
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # Update agent
            if agent.total_steps >= config.WARMUP_STEPS:
                for _ in range(config.GRADIENT_STEPS):
                    actor_loss, critic_loss, alpha = agent.update(config.BATCH_SIZE)
                    
                    if actor_loss is not None:
                        actor_losses.append(actor_loss)
                        critic_losses.append(critic_loss)
            
            # Update statistics
            episode_reward += reward
            episode_cost += info['cost_breakdown']['total_cost']
            episode_steps += 1
            agent.total_steps += 1
            
            # Move to next state
            state = next_state
            
            if done:
                break
        
        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_cost = np.mean(episode_costs[-10:])
            alpha_val = agent.alpha.item() if hasattr(agent.alpha, 'item') else agent.alpha
            
            # Calculate % profit vs baseline (no control = $90,692)
            baseline_cost = 90692.0
            profit_pct = ((baseline_cost - episode_cost) / baseline_cost) * 100
            
            print(f"Episode {episode+1}/{config.TOTAL_EPISODES} | "
                  f"Steps: {episode_steps} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg Reward (10): {avg_reward:.2f} | "
                  f"Cost: {episode_cost:.2f} | "
                  f"Profit: {profit_pct:+.2f}% | "
                  f"Alpha: {alpha_val:.4f}")
        
        # Evaluation
        if (episode + 1) % config.EVAL_FREQ == 0:
            eval_reward, eval_cost = evaluate_agent(agent, env, num_episodes=5)
            baseline_cost = 90692.0
            eval_profit_pct = ((baseline_cost - eval_cost) / baseline_cost) * 100
            print(f"\n{'='*60}")
            print(f"Evaluation after Episode {episode+1}")
            print(f"Avg Evaluation Reward: {eval_reward:.2f}")
            print(f"Avg Evaluation Cost: {eval_cost:.2f}")
            print(f"Avg Profit: {eval_profit_pct:+.2f}%")
            print(f"{'='*60}\n")
        
        # Save model
        if (episode + 1) % config.SAVE_FREQ == 0:
            model_path = os.path.join(config.MODEL_DIR, f"sac_episode_{episode+1}.pt")
            agent.save(model_path)
            print(f"Model saved to {model_path}")
    
    # Save final model
    final_model_path = os.path.join(config.MODEL_DIR, "sac_final.pt")
    agent.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Plot training curves
    plot_training_curves(episode_rewards, episode_costs, actor_losses, critic_losses, config.LOG_DIR)
    
    return agent, episode_rewards, episode_costs

def evaluate_agent(agent, env, num_episodes=5):
    """Evaluate agent performance"""
    eval_rewards = []
    eval_costs = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_cost = 0
        
        for _ in range(env.max_steps):
            action = agent.select_action(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_cost += info['cost_breakdown']['total_cost']
            
            state = next_state
            
            if terminated or truncated:
                break
        
        eval_rewards.append(episode_reward)
        eval_costs.append(episode_cost)
    
    return np.mean(eval_rewards), np.mean(eval_costs)

def plot_training_curves(rewards, costs, actor_losses, critic_losses, save_dir):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # Episode costs
    axes[0, 1].plot(costs)
    axes[0, 1].set_title('Episode Costs')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Total Cost ($)')
    axes[0, 1].grid(True)
    
    # Actor losses
    if len(actor_losses) > 0:
        axes[1, 0].plot(actor_losses)
        axes[1, 0].set_title('Actor Loss')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
    
    # Critic losses
    if len(critic_losses) > 0:
        axes[1, 1].plot(critic_losses)
        axes[1, 1].set_title('Critic Loss')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f'training_curves_{timestamp}.png')
    plt.savefig(save_path, dpi=300)
    print(f"Training curves saved to {save_path}")
    
    plt.close()

def compare_with_qlearning(sac_agent, env, qlearning_results_path):
    """Compare SAC performance with Q-Learning baseline"""
    print("\n" + "="*60)
    print("Comparing SAC with Q-Learning Baseline")
    print("="*60)
    
    # Evaluate SAC
    sac_reward, sac_cost = evaluate_agent(sac_agent, env, num_episodes=10)
    
    print(f"SAC Performance:")
    print(f"  Average Reward: {sac_reward:.2f}")
    print(f"  Average Cost: {sac_cost:.2f}")
    
    # Load Q-Learning results if available
    try:
        import pandas as pd
        qlearning_data = pd.read_csv(qlearning_results_path)
        qlearning_cost = qlearning_data['total_cost'].mean()
        
        print(f"\nQ-Learning Performance:")
        print(f"  Average Cost: {qlearning_cost:.2f}")
        
        improvement = ((qlearning_cost - sac_cost) / qlearning_cost) * 100
        print(f"\nImprovement: {improvement:.2f}%")
        
    except Exception as e:
        print(f"\nCould not load Q-Learning results: {e}")
    
    print("="*60)

if __name__ == "__main__":
    # Train SAC agent
    agent, rewards, costs = train_sac()
    
    # Optional: Compare with Q-Learning
    # qlearning_results_path = r'..\Code_QLearning\AnalysisOfImplementation_v7.csv'
    # if os.path.exists(qlearning_results_path):
    #     compare_with_qlearning(agent, env, qlearning_results_path)
    
    print("\nTraining completed successfully!")
