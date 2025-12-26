# TD3 Pipeline (Microgrid)

## Run Training

```bash
python TD3/td3_main.py
```

- Checkpoints: `./models/td3_microgrid/`
- Logs/curves: `./logs/td3_microgrid/`
- Episodes: `TD3_EPISODES` env var overrides the default 800.

## Configuration

Edit `TD3/td3_config.py` for hyperparameters (noise scales, policy delay, warmup steps) and logging/model paths. TD3 reuses the SAC environment and data loader via `SAC.microgrid_env` and `SAC.sac_config` (shared data path and action/state definitions).

## Evaluation

Use the helper in `td3_main.py`:

```python
from TD3.td3_config import TD3Config
from TD3.td3_agent import TD3Agent
from SAC.microgrid_env import MicrogridEnv

config = TD3Config()
env = MicrogridEnv(config.DATA_PATH, config)
agent = TD3Agent(config)
agent.load('./models/td3_microgrid/td3_final.pt')

state, _ = env.reset()
reward_sum = 0
for _ in range(env.max_steps):
    action = agent.select_action(state, noise=0.0)
    state, reward, terminated, truncated, info = env.step(action)
    reward_sum += reward
    if terminated or truncated:
        break
print(f"Episode reward: {reward_sum:.2f}")
```
