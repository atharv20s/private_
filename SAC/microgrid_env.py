# FIXED: Gymnasium Environment for Multi-Microgrid Energy Management
# This version properly accounts for ESS/EV actions in the energy balance

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import sys
import os

class MicrogridEnv(gym.Env):
    """Custom Gym Environment for 8-Microgrid Energy Management System"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, data_path, config):
        super(MicrogridEnv, self).__init__()
        
        self.config = config
        self.data_path = data_path
        
        # Check if data file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"\n{'='*70}\n"
                f"ERROR: Input data file not found: {data_path}\n"
                f"Please create the input data file by running:\n"
                f"  python create_sample_data.py\n\n"
                f"This will generate sample data with the required columns:\n"
                f"  - Load data: IL1, IL2, CL3, CL4, SL5, SL6, SL7, CPL8\n"
                f"  - PV data: IP1, IP2, CP3, CP4, SP5, SP6, SP7, CPP8\n"
                f"  - Prices: GBP, MBP, MSP, GSP, CHP_Cost\n"
                f"  - EV Availability: EV_Avail\n"
                f"{'='*70}"
            )
        
        # Load data
        self.data = pd.read_csv(data_path)
        
        # Verify required columns
        required_columns = [
            'IL1', 'IP1', 'IL2', 'IP2',
            'CL3', 'CP3', 'CL4', 'CP4',
            'SL5', 'SP5', 'SL6', 'SP6', 'SL7', 'SP7',
            'CPL8', 'CPP8',
            'GBP', 'MBP', 'MSP', 'GSP', 'CHP_Cost', 'EV_Avail'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(
                f"\n{'='*70}\n"
                f"ERROR: Input data file is missing required columns:\n"
                f"  {missing_columns}\n\n"
                f"Current columns: {list(self.data.columns)}\n\n"
                f"Please run: python create_sample_data.py\n"
                f"to generate a properly formatted input file.\n"
                f"{'='*70}"
            )
        
        self.max_steps = len(self.data)
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(config.ACTION_DIM,), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(config.STATE_DIM,),
            dtype=np.float32
        )
        
        # Track actions for each step (for debugging)
        self.action_history = []
        
        # Reset to initialize
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Initialize ESS/EV to 50% capacity
        self.ESS_EV_status = {
            "com1": 50.0,  # 50% of ESS_MAX (100)
            "com2": 50.0,
            "sd1": 8.0,    # 50% of EV_MAX (16)
            "sd2": 8.0,
            "sd3": 8.0,
            "camp": 50.0
        }
        
        self.current_step = 0
        self.episode_cost = 0.0
        self.episode_energy_traded = 0.0
        self.action_history = []
        
        # NEW: Track ESS/EV power flow at each step
        self.ess_ev_power_flow = {
            "com1": 0.0,
            "com2": 0.0,
            "sd1": 0.0,
            "sd2": 0.0,
            "sd3": 0.0,
            "camp": 0.0
        }
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """Execute one time step within the environment"""
        # Clip actions to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Store action for analysis
        self.action_history.append(action.copy())
        
        # CRITICAL FIX: Apply actions FIRST and track power flow
        ess_ev_power = self._apply_actions(action)
        
        # CRITICAL FIX: Calculate energy balance INCLUDING ESS/EV contributions
        energy_data = self._calculate_energy_balance(ess_ev_power)
        
        # Calculate reward (negative cost)
        reward, cost_breakdown = self._calculate_reward(energy_data)
        
        # Update episode statistics
        self.episode_cost += cost_breakdown['total_cost']
        self.episode_energy_traded += energy_data['total_traded']
        
        # Move to next time step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Get next observation
        if not terminated:
            observation = self._get_observation()
        else:
            observation = np.zeros(self.config.STATE_DIM, dtype=np.float32)
        
        # Additional info for logging
        info = {
            'cost_breakdown': cost_breakdown,
            'energy_data': energy_data,
            'ESS_EV_status': self.ESS_EV_status.copy(),
            'ess_ev_power': ess_ev_power.copy(),
            'step': self.current_step
        }
        
        return observation, reward, terminated, truncated, info
    
    def _apply_actions(self, action):
        """
        Apply charging/discharging actions to ESS and EV
        
        CRITICAL FIX: Returns the POWER FLOW (kW) not just state updates
        Positive = charging (consuming power), Negative = discharging (providing power)
        """
        action_mapping = {
            0: "com1", 1: "com2", 2: "sd1", 
            3: "sd2", 4: "sd3", 5: "camp"
        }
        
        ess_ev_power_flow = {}
        
        for idx, key in action_mapping.items():
            current_level = self.ESS_EV_status[key]
            action_val = action[idx]
            
            # Determine max capacity
            max_capacity = self.config.ESS_MAX if key in ["com1", "com2", "camp"] else self.config.EV_MAX
            
            # Calculate POWER FLOW (kW) based on action
            # action_val in [-1, 1] represents percentage of max charge/discharge rate
            # Use configurable action scale (default 5% instead of 10%)
            max_power_rate = max_capacity * self.config.ACTION_SCALE
            
            if action_val > 0:  # Charging (consuming power from grid)
                # Power consumed (positive = deficit increase)
                power_consumed = action_val * max_power_rate
                
                # Update battery level with efficiency losses
                energy_stored = power_consumed * self.config.N_C
                new_level = min(current_level + energy_stored, max_capacity)
                
                # Actual power consumed (considering battery limits)
                actual_power = (new_level - current_level) / self.config.N_C
                ess_ev_power_flow[key] = actual_power  # Positive = consuming
                
            elif action_val < 0:  # Discharging (providing power to grid)
                # Power provided (negative = deficit decrease)
                power_provided = abs(action_val) * max_power_rate
                
                # Update battery level with efficiency losses
                energy_released = power_provided * self.config.N_D
                new_level = max(current_level - energy_released, 0.0)
                
                # Actual power provided (considering battery limits)
                actual_power = (current_level - new_level) / self.config.N_D
                ess_ev_power_flow[key] = -actual_power  # Negative = providing
                
            else:  # Idle
                new_level = current_level
                ess_ev_power_flow[key] = 0.0
            
            self.ESS_EV_status[key] = new_level
        
        return ess_ev_power_flow
    
    def _calculate_energy_balance(self, ess_ev_power):
        """
        CRITICAL FIX: Calculate energy surplus/deficit INCLUDING ESS/EV contributions
        
        ess_ev_power: dict with power flow for each ESS/EV
          Positive = charging (increases deficit)
          Negative = discharging (decreases deficit / increases surplus)
        """
        i = self.current_step
        
        # Read prices and costs
        gbp = self.data.loc[i, "GBP"]
        mbp = self.data.loc[i, "MBP"]
        msp = self.data.loc[i, "MSP"]
        gsp = self.data.loc[i, "GSP"]
        chp_cost = self.data.loc[i, "CHP_Cost"]
        
        total_deficit = 0.0
        total_surplus = 0.0
        mg_data = {}
        
        # Industry MGs (1, 2) - use CHP optimization (no ESS/EV)
        for mg_id in [1, 2]:
            load = self.data.loc[i, f"IL{mg_id}"]
            pv = self.data.loc[i, f"IP{mg_id}"]
            deficit = max(0, load - pv)
            surplus = max(0, pv - load)
            
            total_deficit += deficit
            total_surplus += surplus
            mg_data[f'ind{mg_id}'] = {'deficit': deficit, 'surplus': surplus}
        
        # Community MGs (3, 4) - WITH ESS
        for mg_id, key in [(3, 'com1'), (4, 'com2')]:
            load = self.data.loc[i, f"CL{mg_id}"]
            pv = self.data.loc[i, f"CP{mg_id}"]
            
            # Base deficit/surplus from load-PV
            base_deficit = max(0, load - pv)
            base_surplus = max(0, pv - load)
            
            # CRITICAL FIX: Add ESS power flow
            ess_power = ess_ev_power[key]
            
            if ess_power > 0:  # ESS charging (adds to deficit)
                actual_deficit = base_deficit + ess_power
                actual_surplus = max(0, base_surplus - ess_power)
            else:  # ESS discharging (reduces deficit or adds to surplus)
                actual_deficit = max(0, base_deficit + ess_power)  # ess_power is negative
                actual_surplus = base_surplus + abs(ess_power)
            
            total_deficit += actual_deficit
            total_surplus += actual_surplus
            mg_data[key] = {
                'deficit': actual_deficit, 
                'surplus': actual_surplus,
                'ess_power': ess_power
            }
        
        # Single-Dwelling MGs (5, 6, 7) - WITH EV
        ev_available = self.data.loc[i, "EV_Avail"]
        for mg_id, key in [(5, 'sd1'), (6, 'sd2'), (7, 'sd3')]:
            load = self.data.loc[i, f"SL{mg_id}"]
            pv = self.data.loc[i, f"SP{mg_id}"]
            
            # Base deficit/surplus
            base_deficit = max(0, load - pv)
            base_surplus = max(0, pv - load)
            
            # CRITICAL FIX: Add EV power flow (only if EV available)
            if ev_available > 0:
                ev_power = ess_ev_power[key]
                
                if ev_power > 0:  # EV charging
                    actual_deficit = base_deficit + ev_power
                    actual_surplus = max(0, base_surplus - ev_power)
                else:  # EV discharging (V2G)
                    actual_deficit = max(0, base_deficit + ev_power)
                    actual_surplus = base_surplus + abs(ev_power)
            else:
                actual_deficit = base_deficit
                actual_surplus = base_surplus
            
            total_deficit += actual_deficit
            total_surplus += actual_surplus
            mg_data[key] = {
                'deficit': actual_deficit, 
                'surplus': actual_surplus,
                'ev_power': ess_ev_power[key] if ev_available > 0 else 0.0
            }
        
        # Campus MG (8) - WITH ESS and CHP
        load = self.data.loc[i, "CPL8"]
        pv = self.data.loc[i, "CPP8"]
        
        base_deficit = max(0, load - pv)
        base_surplus = max(0, pv - load)
        
        # CRITICAL FIX: Add ESS power flow
        ess_power = ess_ev_power['camp']
        
        if ess_power > 0:  # ESS charging
            actual_deficit = base_deficit + ess_power
            actual_surplus = max(0, base_surplus - ess_power)
        else:  # ESS discharging
            actual_deficit = max(0, base_deficit + ess_power)
            actual_surplus = base_surplus + abs(ess_power)
        
        total_deficit += actual_deficit
        total_surplus += actual_surplus
        mg_data['camp'] = {
            'deficit': actual_deficit, 
            'surplus': actual_surplus,
            'ess_power': ess_power
        }
        
        return {
            'total_deficit': total_deficit,
            'total_surplus': total_surplus,
            'total_traded': min(total_deficit, total_surplus),
            'mg_data': mg_data,
            'prices': {'gbp': gbp, 'mbp': mbp, 'msp': msp, 'gsp': gsp},
            'chp_cost': chp_cost
        }
    
    def _calculate_reward(self, energy_data):
        """Calculate reward with shaping for better learning"""
        total_deficit = energy_data['total_deficit']
        total_surplus = energy_data['total_surplus']
        prices = energy_data['prices']
        mg_data = energy_data['mg_data']
        
        # Cost components
        grid_purchase_cost = 0.0
        grid_sale_revenue = 0.0
        constraint_penalty = 0.0
        
        # Trading cost calculation
        if total_deficit > total_surplus:
            # Need to buy from grid
            unmet_deficit = total_deficit - total_surplus
            grid_purchase_cost = unmet_deficit * prices['mbp']
            grid_sale_revenue = 0.0
        else:
            # Surplus exceeds deficit, sell excess to grid
            excess_surplus = total_surplus - total_deficit
            grid_sale_revenue = excess_surplus * prices['gsp']
            grid_purchase_cost = 0.0
        
        # Penalty for ESS/EV constraint violations
        for key, level in self.ESS_EV_status.items():
            max_cap = self.config.ESS_MAX if key in ['com1', 'com2', 'camp'] else self.config.EV_MAX
            
            if level < 0 or level > max_cap:
                constraint_penalty += abs(self.config.CONSTRAINT_PENALTY)
        
        # === REWARD SHAPING FOR BETTER LEARNING ===
        
        # 1. Base cost (main objective)
        net_energy_cost = grid_purchase_cost - grid_sale_revenue
        
        # 2. Self-sufficiency bonus (reduce grid dependency)
        total_load_equivalent = total_deficit + total_surplus  # Approximate total energy flow
        if total_load_equivalent > 0:
            self_sufficiency_ratio = min(total_deficit, total_surplus) / total_load_equivalent
            self_sufficiency_bonus = self_sufficiency_ratio * 100  # Up to $100 bonus
        else:
            self_sufficiency_bonus = 0
        
        # 3. ESS utilization bonus (reward for actually using storage)
        ess_utilization_bonus = 0
        for key in ['com1', 'com2', 'camp']:
            if key in mg_data and 'ess_power' in mg_data[key]:
                ess_power = abs(mg_data[key]['ess_power'])
                # Small bonus for using ESS (encourages exploration)
                ess_utilization_bonus += min(ess_power * 0.1, 10)  # Max $10 per ESS
        
        # 4. Price-responsive bonus (reward charging when cheap, discharging when expensive)
        price_responsiveness_bonus = 0
        mbp = prices['mbp']
        mbp_range = self.config.MBP_MAX - self.config.MBP_MIN
        price_normalized = (mbp - self.config.MBP_MIN) / mbp_range  # [0, 1]
        
        for key in ['com1', 'com2', 'sd1', 'sd2', 'sd3', 'camp']:
            if key in mg_data and 'ess_power' in mg_data[key]:
                ess_power = mg_data[key]['ess_power']
                # Reward charging when price is low, discharging when price is high
                if ess_power > 0 and price_normalized < 0.4:  # Charging during low price
                    price_responsiveness_bonus += ess_power * 0.5
                elif ess_power < 0 and price_normalized > 0.6:  # Discharging during high price
                    price_responsiveness_bonus += abs(ess_power) * 0.5
        
        # Total cost
        total_cost = net_energy_cost + constraint_penalty
        
        # Original reward: negative cost (lower cost = higher reward)
        reward = -total_cost / 1000.0  # Scale to ~-90 range
        
        cost_breakdown = {
            'total_cost': total_cost,
            'grid_purchase': grid_purchase_cost,
            'grid_revenue': grid_sale_revenue,
            'constraint_penalty': constraint_penalty,
            'net_energy_cost': net_energy_cost,
            'self_sufficiency_bonus': self_sufficiency_bonus,
            'ess_utilization_bonus': ess_utilization_bonus,
            'price_responsiveness_bonus': price_responsiveness_bonus,
            'shaped_reward': reward
        }
        
        return reward, cost_breakdown
    
    def _get_observation(self):
        """Get current state observation"""
        i = min(self.current_step, len(self.data) - 1)
        
        # Build state dictionary
        state_dict = {
            # ESS/EV levels
            'com1_ess': self.ESS_EV_status['com1'],
            'com2_ess': self.ESS_EV_status['com2'],
            'camp_ess': self.ESS_EV_status['camp'],
            'sd1_ev': self.ESS_EV_status['sd1'],
            'sd2_ev': self.ESS_EV_status['sd2'],
            'sd3_ev': self.ESS_EV_status['sd3'],
            
            # Loads for all 8 MGs
            'load_mg1': self.data.loc[i, 'IL1'],
            'load_mg2': self.data.loc[i, 'IL2'],
            'load_mg3': self.data.loc[i, 'CL3'],
            'load_mg4': self.data.loc[i, 'CL4'],
            'load_mg5': self.data.loc[i, 'SL5'],
            'load_mg6': self.data.loc[i, 'SL6'],
            'load_mg7': self.data.loc[i, 'SL7'],
            'load_mg8': self.data.loc[i, 'CPL8'],
            
            # PV generation for all 8 MGs
            'pv_mg1': self.data.loc[i, 'IP1'],
            'pv_mg2': self.data.loc[i, 'IP2'],
            'pv_mg3': self.data.loc[i, 'CP3'],
            'pv_mg4': self.data.loc[i, 'CP4'],
            'pv_mg5': self.data.loc[i, 'SP5'],
            'pv_mg6': self.data.loc[i, 'SP6'],
            'pv_mg7': self.data.loc[i, 'SP7'],
            'pv_mg8': self.data.loc[i, 'CPP8'],
            
            # Prices
            'gbp': self.data.loc[i, 'GBP'],
            'mbp': self.data.loc[i, 'MBP'],
            'msp': self.data.loc[i, 'MSP'],
            'gsp': self.data.loc[i, 'GSP'],
            
            # Time features
            'time_of_day': i % 24,
            'day_of_week': (i // 24) % 7,
            
            # Energy balance (will be calculated)
            'total_deficit': 0.0,
            'total_surplus': 0.0,
            'stress': 0.5
        }
        
        # Calculate current energy balance (with zero power flow for observation)
        zero_power = {key: 0.0 for key in ['com1', 'com2', 'sd1', 'sd2', 'sd3', 'camp']}
        energy_data = self._calculate_energy_balance(zero_power)
        
        state_dict['total_deficit'] = energy_data['total_deficit']
        state_dict['total_surplus'] = energy_data['total_surplus']
        state_dict['stress'] = energy_data['total_surplus'] / (energy_data['total_deficit'] + 1e-6)
        
        # Normalize state using config function
        observation = np.array(self.config.normalize_state(state_dict), dtype=np.float32)
        
        return observation
    
    def render(self):
        """Render the environment (optional)"""
        print(f"Step: {self.current_step}, Episode Cost: {self.episode_cost:.2f}")
        print(f"ESS/EV Status: {self.ESS_EV_status}")
        if self.action_history:
            print(f"Last Action: {self.action_history[-1]}")