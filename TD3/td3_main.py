# TD3 Training Script for Multi-Microgrid Energy Management

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from td3_config import TD3Config
from td3_agent import TD3Agent
from SAC.microgrid_env import MicrogridEnv


def train_td3():
    config = TD3Config()

    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    env = MicrogridEnv(config.DATA_PATH, config)
    agent = TD3Agent(config)

    episode_rewards = []
    episode_costs = []
    actor_losses = []
    critic_losses = []

    print("=" * 60)
    print("Starting TD3 Training for Multi-Microgrid Energy Management")
    print("=" * 60)
    print(f"State Dimension: {config.STATE_DIM}")
    print(f"Action Dimension: {config.ACTION_DIM}")
    print(f"Total Episodes: {config.TOTAL_EPISODES}")
    print(f"Device: {config.DEVICE}")
    print(f"Action Scale: {config.ACTION_SCALE * 100:.1f}% of capacity per step")
    print(f"Warmup Steps: {config.WARMUP_STEPS}")
    print("=" * 60)

    for episode in range(config.TOTAL_EPISODES):
        state, info = env.reset()
        episode_reward = 0.0
        episode_cost = 0.0
        episode_steps = 0

        for step in range(config.MAX_STEPS_PER_EPISODE):
            if agent.total_steps < config.WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, noise=config.ACTION_NOISE)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.add(state, action, reward, next_state, float(done))

            if agent.total_steps >= config.WARMUP_STEPS:
                for _ in range(config.GRADIENT_STEPS):
                    actor_loss, critic_loss = agent.train(config.BATCH_SIZE)
                    if actor_loss is not None:
                        actor_losses.append(actor_loss)
                    if critic_loss is not None:
                        critic_losses.append(critic_loss)

            episode_reward += reward
            episode_cost += info["cost_breakdown"]["total_cost"]
            episode_steps += 1
            agent.total_steps += 1
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_cost = np.mean(episode_costs[-10:])
            # Calculate % profit vs baseline (no control = $90,692)
            baseline_cost = 90692.0
            profit_pct = ((baseline_cost - episode_cost) / baseline_cost) * 100
            
            print(
                f"Episode {episode + 1}/{config.TOTAL_EPISODES} | "
                f"Steps: {episode_steps} | "
                f"Reward: {episode_reward:.2f} | "
                f"Avg Reward (10): {avg_reward:.2f} | "
                f"Cost: {episode_cost:.2f} | "
                f"Profit: {profit_pct:+.2f}%"
            )

        if (episode + 1) % config.EVAL_FREQ == 0:
            eval_reward, eval_cost = evaluate_agent(agent, env, num_episodes=3)
            baseline_cost = 90692.0
            eval_profit_pct = ((baseline_cost - eval_cost) / baseline_cost) * 100
            print("\n" + "=" * 60)
            print(f"Evaluation after Episode {episode + 1}")
            print(f"Avg Evaluation Reward: {eval_reward:.2f}")
            print(f"Avg Evaluation Cost: {eval_cost:.2f}")
            print(f"Avg Profit: {eval_profit_pct:+.2f}%")
            print("=" * 60 + "\n")

        if (episode + 1) % config.SAVE_FREQ == 0:
            model_path = os.path.join(config.MODEL_DIR, f"td3_episode_{episode + 1}.pt")
            agent.save(model_path)
            print(f"Model saved to {model_path}")

    final_model_path = os.path.join(config.MODEL_DIR, "td3_final.pt")
    agent.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")

    plot_training_curves(episode_rewards, episode_costs, actor_losses, critic_losses, config.LOG_DIR)
    return agent, episode_rewards, episode_costs


def evaluate_agent(agent, env, num_episodes=3):
    rewards = []
    costs = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_cost = 0.0
        for _ in range(env.max_steps):
            action = agent.select_action(state, noise=0.0)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_cost += info["cost_breakdown"]["total_cost"]
            state = next_state
            if terminated or truncated:
                break
        rewards.append(episode_reward)
        costs.append(episode_cost)
    return float(np.mean(rewards)), float(np.mean(costs))


def plot_training_curves(rewards, costs, actor_losses, critic_losses, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(rewards)
    axes[0, 0].set_title("Episode Rewards")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Total Reward")
    axes[0, 0].grid(True)

    axes[0, 1].plot(costs)
    axes[0, 1].set_title("Episode Costs")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Total Cost ($)")
    axes[0, 1].grid(True)

    if len(actor_losses) > 0:
        axes[1, 0].plot(actor_losses)
        axes[1, 0].set_title("Actor Loss")
        axes[1, 0].set_xlabel("Training Step")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].grid(True)

    if len(critic_losses) > 0:
        axes[1, 1].plot(critic_losses)
        axes[1, 1].set_title("Critic Loss")
        axes[1, 1].set_xlabel("Training Step")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].grid(True)

    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"training_curves_td3_{timestamp}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Training curves saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    train_td3()
