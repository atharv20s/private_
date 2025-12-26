# Visualize SAC Training Progress and Compare with Baselines

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_learning_curves(log_dir='./logs/sac_microgrid', prefix='training_curves_', label='SAC'):
    """Create visualization of training progress for a given algorithm."""

    training_images = [
        f for f in os.listdir(log_dir)
        if f.startswith(prefix) and f.endswith('.png')
    ]

    if not training_images:
        print(f"No training curves found in {log_dir}. Training may still be in progress.")
        return

    latest_image = sorted(training_images)[-1]
    print(f"Latest {label} training curves: {os.path.join(log_dir, latest_image)}")
    
    # Load baseline comparison
    baseline_file = './logs/baseline_comparison.csv'
    if os.path.exists(baseline_file):
        baseline_df = pd.read_csv(baseline_file)
        print("\n" + "="*80)
        print("BASELINE COMPARISON")
        print("="*80)
        print(baseline_df.to_string(index=False))
        
        best_baseline = baseline_df.loc[baseline_df['Baseline'] == 'Optimal Threshold', 'Cost'].values[0]
        theoretical_min = baseline_df.loc[baseline_df['Baseline'] == 'Theoretical Min', 'Cost'].values[0]
        no_action = baseline_df.loc[baseline_df['Baseline'] == 'No Action', 'Cost'].values[0]
        
        print(f"\n🎯 Target Performance:")
        print(f"   Theoretical Minimum: ${theoretical_min:.2f}")
        print(f"   Best Baseline: ${best_baseline:.2f}")
        print(f"   Do Nothing: ${no_action:.2f}")
        print(f"   Available Savings: ${no_action - theoretical_min:.2f}")
    else:
        print("Baseline comparison not found. Run test_baseline.py first.")
        best_baseline = 89400
        theoretical_min = 89274
        no_action = 90692

def analyze_hourly_performance(
    model_path='./models/sac_microgrid/sac_final.pt',
    config_cls=None,
    agent_cls=None,
    label='SAC'
):
    """Analyze hourly performance for a trained agent."""

    if config_cls is None:
        from sac_config import SACConfig as config_cls  # type: ignore
    if agent_cls is None:
        from sac_agent import SACAgent as agent_cls  # type: ignore
    from microgrid_env import MicrogridEnv

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Training may still be in progress.")
        return

    print("\n" + "="*80)
    print(f"HOURLY PERFORMANCE ANALYSIS ({label})")
    print("="*80)

    config = config_cls()
    env = MicrogridEnv(config.DATA_PATH, config)
    agent = agent_cls(config)

    try:
        agent.load(model_path)
        print(f"✓ Loaded model: {model_path}")
    except Exception:
        print("✗ Could not load model")
        return
    
    # Run one episode and track hourly costs
    state, _ = env.reset()
    hourly_data = []
    
    for hour in range(24):
        # Get SAC action
        action = agent.select_action(state, deterministic=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Get data for this hour
        mbp = env.data.loc[hour, 'MBP']
        loads = [env.data.loc[hour, f'IL{i}'] for i in [1, 2]] + \
                [env.data.loc[hour, f'CL{i}'] for i in [3, 4]] + \
                [env.data.loc[hour, f'SL{i}'] for i in [5, 6, 7]] + \
                [env.data.loc[hour, f'CPL8']]
        pvs = [env.data.loc[hour, f'IP{i}'] for i in [1, 2]] + \
              [env.data.loc[hour, f'CP{i}'] for i in [3, 4]] + \
              [env.data.loc[hour, f'SP{i}'] for i in [5, 6, 7]] + \
              [env.data.loc[hour, f'CPP8']]
        
        total_load = sum(loads)
        total_pv = sum(pvs)
        
        hourly_data.append({
            'Hour': hour + 1,
            'Load': total_load,
            'PV': total_pv,
            'Net_Load': total_load - total_pv,
            'MBP': mbp,
            'Cost': info['cost_breakdown']['total_cost'],
            'Grid_Purchase': info['cost_breakdown']['grid_purchase'],
            'Deficit': info['energy_data']['total_deficit'],
            'Surplus': info['energy_data']['total_surplus']
        })
        
        state = next_state
        if terminated or truncated:
            break
    
    # Create DataFrame
    df = pd.DataFrame(hourly_data)
    
    # Find most expensive hours
    print(f"\n📊 Most Expensive Hours (Top 5):")
    print(df.nlargest(5, 'Cost')[['Hour', 'Load', 'PV', 'MBP', 'Cost']].to_string(index=False))
    
    print(f"\n📊 Cheapest Hours (Top 5):")
    print(df.nsmallest(5, 'Cost')[['Hour', 'Load', 'PV', 'MBP', 'Cost']].to_string(index=False))
    
    print(f"\n📊 High Price Hours (MBP > $6.50):")
    high_price = df[df['MBP'] > 6.5]
    if len(high_price) > 0:
        print(high_price[['Hour', 'Load', 'PV', 'MBP', 'Deficit', 'Cost']].to_string(index=False))
    else:
        print("None")
    
    print(f"\n📊 Low Price Hours (MBP < $5.50):")
    low_price = df[df['MBP'] < 5.5]
    if len(low_price) > 0:
        print(low_price[['Hour', 'Load', 'PV', 'MBP', 'Deficit', 'Cost']].to_string(index=False))
    else:
        print("None")
    
    # Summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"   Total Daily Cost: ${df['Cost'].sum():.2f}")
    print(f"   Avg Hourly Cost: ${df['Cost'].mean():.2f}")
    print(f"   Std Hourly Cost: ${df['Cost'].std():.2f}")
    print(f"   Peak Hour Cost: ${df['Cost'].max():.2f} (Hour {df.loc[df['Cost'].idxmax(), 'Hour']:.0f})")
    
    # Save detailed results
    df.to_csv('./logs/hourly_analysis.csv', index=False)
    print(f"\n✓ Detailed hourly data saved to ./logs/hourly_analysis.csv")
    
    # Create hourly cost plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(df['Hour'], df['Cost'], 'b-o', label='Hourly Cost', linewidth=2)
    plt.xlabel('Hour of Day')
    plt.ylabel('Cost ($)')
    plt.title('Hourly Energy Cost')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(df['Hour'], df['MBP'], 'r-s', label='Market Buy Price', linewidth=2)
    plt.xlabel('Hour of Day')
    plt.ylabel('Price ($/kWh)')
    plt.title('Market Price by Hour')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./logs/hourly_analysis.png', dpi=150)
    print(f"✓ Hourly analysis plot saved to ./logs/hourly_analysis.png")
    plt.close()

def compare_with_baseline_detailed(
    model_path='./models/sac_microgrid/sac_final.pt',
    config_cls=None,
    agent_cls=None,
    label='SAC'
):
    """Compare an RL agent with baseline policies hour-by-hour."""
    if config_cls is None:
        from sac_config import SACConfig as config_cls  # type: ignore
    if agent_cls is None:
        from sac_agent import SACAgent as agent_cls  # type: ignore
    from microgrid_env import MicrogridEnv

    print("\n" + "="*80)
    print(f"DETAILED {label} vs BASELINE COMPARISON")
    print("="*80)

    config = config_cls()
    env = MicrogridEnv(config.DATA_PATH, config)

    if not os.path.exists(model_path):
        print("Final model not found. Training may still be in progress.")
        return

    agent = agent_cls(config)
    agent.load(model_path)
    
    # Get SAC cost
    state, _ = env.reset()
    sac_cost = 0
    for _ in range(24):
        action = agent.select_action(state, deterministic=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        sac_cost += info['cost_breakdown']['total_cost']
        state = next_state
        if terminated or truncated:
            break
    
    # Get No Action cost
    state, _ = env.reset()
    no_action_cost = 0
    for _ in range(24):
        action = np.zeros(6)
        next_state, reward, terminated, truncated, info = env.step(action)
        no_action_cost += info['cost_breakdown']['total_cost']
        state = next_state
        if terminated or truncated:
            break
    
    # Get Optimal Threshold cost
    median_price = env.data['MBP'].median()
    state, _ = env.reset()
    optimal_cost = 0
    for step in range(24):
        mbp = env.data.loc[step, 'MBP']
        if mbp < median_price - 0.5:
            action = np.ones(6) * 0.8
        elif mbp < median_price:
            action = np.ones(6) * 0.4
        elif mbp > median_price + 0.5:
            action = np.ones(6) * -0.8
        elif mbp > median_price:
            action = np.ones(6) * -0.4
        else:
            action = np.zeros(6)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        optimal_cost += info['cost_breakdown']['total_cost']
        state = next_state
        if terminated or truncated:
            break
    
    print(f"\n📊 Final Comparison:")
    print(f"   SAC Agent:           ${sac_cost:.2f}")
    print(f"   Optimal Threshold:   ${optimal_cost:.2f} (gap: ${sac_cost - optimal_cost:+.2f})")
    print(f"   No Action:           ${no_action_cost:.2f} (gap: ${sac_cost - no_action_cost:+.2f})")
    
    savings = no_action_cost - sac_cost
    max_savings = no_action_cost - optimal_cost
    efficiency = (savings / max_savings * 100) if max_savings > 0 else 0
    
    print(f"\n🎯 Performance Metrics:")
    print(f"   Savings vs Do Nothing: ${savings:.2f} ({savings/no_action_cost*100:.2f}%)")
    print(f"   Maximum Possible Savings: ${max_savings:.2f}")
    print(f"   SAC Efficiency: {efficiency:.1f}% of maximum savings captured")

if __name__ == "__main__":
    print("="*80)
    print("SAC TRAINING ANALYSIS & VISUALIZATION")
    print("="*80)

    plot_learning_curves()
    analyze_hourly_performance()
    compare_with_baseline_detailed()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
