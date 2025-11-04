# Multi-Microgrid Energy Management using Reinforcement Learning

This repository implements and compares two reinforcement learning approaches for energy management in multi-microgrid systems:
1. Q-Learning (tabular method)
2. Soft Actor-Critic (SAC, deep RL method)

## Directory Structure

```
IEEE_RL/
├── Code_QLearning/         # Q-Learning implementation
│   ├── main_v7.py         # Main entry point
│   ├── Q_Learning_v7.py   # Q-Learning algorithm
│   ├── code_v7.py         # Helper functions
│   └── settings*.py       # Configuration files
├── SAC/                   # SAC implementation
│   ├── sac_main.py       # Main training script
│   ├── sac_agent.py      # SAC agent implementation
│   ├── sac_config.py     # Hyperparameters
│   └── microgrid_env.py  # Environment wrapper
├── artifacts/            # Generated visualizations & results
├── models/              # Saved model checkpoints
└── logs/               # Training logs and metrics
```

## Installation

1. Create a Python virtual environment:
```bash
python -m venv .venv
```

2. Activate the environment:
```bash
# Windows
.\.venv\Scripts\Activate.ps1
# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Q-Learning Implementation

```bash
# Run Q-Learning training
python ./Code_QLearning/main_v7.py
```

### SAC Implementation

```bash
# Train SAC agent
python ./SAC/sac_main.py

# Evaluate trained SAC agent
python ./SAC/evaluate_sac.py

# Compare Q-Learning and SAC performance
python ./SAC/compare_qlearning_sac.py
```

## Results

See `comparison_Qlearning_vs_SAC.md` for detailed performance analysis and comparison between the two approaches. Key findings:

- SAC achieves 2.0% better cost reduction than Q-Learning
- SAC trains 4x faster (30 min vs 120 min)
- SAC requires 2x fewer episodes to converge
- SAC enables better asset utilization through continuous control

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details.