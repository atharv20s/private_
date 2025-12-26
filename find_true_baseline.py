"""
Find the TRUE baseline cost for microgrid operation without RL optimization.
This determines if your RL agents are actually improving performance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SAC'))

import numpy as np
from SAC.microgrid_env import MicrogridEnv
from SAC.sac_config import SACConfig

def test_no_control(env, num_episodes=50):
    """Baseline: No charging/discharging (all zeros)"""
    costs = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            # No control - all actions = 0
            action = np.zeros(env.action_space.shape[0])
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # Get final episode cost
        episode_cost = env.episode_cost
        costs.append(episode_cost)
    
    return np.mean(costs), np.std(costs), costs

def test_random_control(env, num_episodes=50):
    """Baseline: Random actions within valid range"""
    costs = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            # Random actions
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # Get final episode cost
        episode_cost = env.episode_cost
        costs.append(episode_cost)
    
    return np.mean(costs), np.std(costs), costs

def test_simple_rule(env, num_episodes=50):
    """Baseline: Charge at low prices, discharge at high prices (simple threshold)"""
    costs = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            # Simple rule: if price < median, charge at 0.3, else discharge at -0.3
            current_price = state[18]  # Assuming price is at index 18
            
            if current_price < 150:  # Low price threshold
                action = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
            else:  # High price
                action = np.array([-0.3, -0.3, -0.3, -0.3, -0.3, -0.3])
            
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # Get final episode cost
        episode_cost = env.episode_cost
        costs.append(episode_cost)
    
    return np.mean(costs), np.std(costs), costs

def main():
    config = SACConfig()
    data_path = config.DATA_PATH
    env = MicrogridEnv(data_path, config)
    
    print("=" * 70)
    print("FINDING TRUE BASELINE COST")
    print("=" * 70)
    print(f"Running {50} episodes for each baseline strategy...\n")
    
    # Test 1: No control
    print("🔹 Strategy 1: NO CONTROL (all actions = 0)")
    no_control_mean, no_control_std, _ = test_no_control(env, 50)
    print(f"   Mean Cost: ${no_control_mean:,.2f}")
    print(f"   Std Dev:   ${no_control_std:,.2f}\n")
    
    # Test 2: Random control
    print("🔹 Strategy 2: RANDOM ACTIONS")
    random_mean, random_std, _ = test_random_control(env, 50)
    print(f"   Mean Cost: ${random_mean:,.2f}")
    print(f"   Std Dev:   ${random_std:,.2f}\n")
    
    # Test 3: Simple rule
    print("🔹 Strategy 3: SIMPLE RULE (charge low price, discharge high)")
    rule_mean, rule_std, _ = test_simple_rule(env, 50)
    print(f"   Mean Cost: ${rule_mean:,.2f}")
    print(f"   Std Dev:   ${rule_std:,.2f}\n")
    
    # Determine best baseline
    worst_baseline = max(no_control_mean, random_mean, rule_mean)
    best_baseline = min(no_control_mean, random_mean, rule_mean)
    
    print("=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)
    print(f"Worst (Conservative) Baseline: ${worst_baseline:,.2f}")
    print(f"Best (Optimistic) Baseline:    ${best_baseline:,.2f}")
    print(f"Recommended Baseline:          ${no_control_mean:,.2f} (no control)\n")
    
    # Compare with RL results
    print("=" * 70)
    print("COMPARISON WITH YOUR RL AGENTS")
    print("=" * 70)
    sac_cost = 88405.28
    td3_cost = 91607.49
    
    print(f"SAC Cost:  ${sac_cost:,.2f}")
    print(f"TD3 Cost:  ${td3_cost:,.2f}\n")
    
    # Calculate savings
    for name, baseline in [("Conservative", worst_baseline), 
                           ("Realistic", no_control_mean),
                           ("Optimistic", best_baseline)]:
        sac_savings = baseline - sac_cost
        td3_savings = baseline - td3_cost
        sac_pct = (sac_savings / baseline) * 100
        td3_pct = (td3_savings / baseline) * 100
        
        print(f"📊 Against {name} Baseline (${baseline:,.2f}):")
        print(f"   SAC: ${sac_savings:+,.2f} ({sac_pct:+.2f}%)")
        print(f"   TD3: ${td3_savings:+,.2f} ({td3_pct:+.2f}%)\n")
    
    print("=" * 70)
    print("RECOMMENDATION FOR YOUR PROJECT:")
    print("=" * 70)
    print(f"Use baseline = ${no_control_mean:,.2f} (no control scenario)")
    print("This represents a realistic unoptimized microgrid operation.\n")

if __name__ == "__main__":
    main()
