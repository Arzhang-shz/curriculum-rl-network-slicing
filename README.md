# Multi-domain Network Service Placement Optimization Using Curriculum Reinforcement Learning

This repository contains the code and core results for a research project exploring the use of curriculum reinforcement learning (RL) to optimize the placement of network services in a multi-domain environment. The project aims to maximize resource utilization (CPU, memory) and the number of accepted service requests while adhering to service level agreements (SLAs).

## Overview

This project is part of ongoing research in network slicing and resource orchestration.

- **Multi-domain Network Slicing:** The project addresses the challenge of placing network slices across multiple administrative domains, each with its own topology and resource constraints.
- **Curriculum Reinforcement Learning:** A hierarchical RL agent, trained with a curriculum learning approach, learns to efficiently place VNFs across multiple domains.
- **Resource Optimization:** The RL agent optimizes resource utilization (CPU, memory, bandwidth) to maximize the number of accepted service requests.
- **SLA Adherence:** The placement strategy considers end-to-end latency requirements to ensure SLA compliance.

## Repository Structure

```
configs/                # YAML configuration files for curriculum learning stages
    curriculum_stage1.yaml   # Configuration for simple slice requests
    curriculum_stage2.yaml   # Configuration for medium-difficulty slice requests
    curriculum_stage3.yaml   # Configuration for complex slice requests
figures/                # Project Figures
    env_master_AR.eps      # Acceptance ratio plot for the master agent
    env_master_load.eps    # Load plot for the master agent
    reward_1.eps           # Learning curves plot
training/               # Training scripts and related modules
    __init__.py.py
    callbacks.py.py        # Custom callbacks for logging metrics during training
    make_env.py.py         # Functions for creating the RL environment
    plot_utils.py.py       # Utility functions for plotting training metrics
    scheduler.py.py        # Curriculum scheduler for multi-stage training
    train_ppo.py.py        # Main training script for PPO with curriculum learning
requirements.txt        # Python dependencies
README.md               # Project documentation
setup.py                # Installation script
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/multi-domain-CLRL.git
cd multi-domain-CLRL
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the RL Agent

```bash
python training/train_ppo.py --configs configs/curriculum_stage1.yaml configs/curriculum_stage2.yaml configs/curriculum_stage3.yaml
```

## Key Results

The following figures summarize simulation results and provide insights into the performance of the curriculum RL approach.

<p align="center">
  <img src="figures/fig1.png" alt="Multi-domain DRL Slice Placement" width="600"/>
  <br/>
  <em><strong>Fig. 1:</strong> Illustration of slice placement across multiple domains, demonstrating vertical and horizontal training strategies.</em>
</p>

<p align="center">
  <img src="figures/reward_1.jpg" alt="Learning Curves" width="600"/>
  <br/>
  <em><strong>Fig. 2:</strong> Comparison of learning curves using PPO and Curriculum PPO. The red lines indicate stage transitions in curriculum training. Curriculum PPO shows better sample efficiency and final performance.</em>
</p>

<p align="center">
  <img src="figures/env1_AR.jpg" alt="Acceptance Ratio Comparison" width="600"/>
  <br/>
  <em><strong>Fig. 3:</strong> Comparison of acceptance ratio across algorithms (Curriculum PPO, PPO, Greedy) for the master agent. Curriculum PPO achieves the highest acceptance rate.</em>
</p>

<p align="center">
  <img src="figures/env1_load.jpg" alt="Load Comparison" width="600"/>
  <br/>
  <em><strong>Fig. 4:</strong> Load comparison across algorithms for the master agent. Curriculum PPO maintains efficient resource utilization.</em>
</p>




## Citation

If you use this code in your research, please cite the following paper:

[Multi-domain Network Service Placement Optimization Using Curriculum Reinforcement Learning](https://ieeexplore.ieee.org/abstract/document/10329592/)

## License

This project is licensed under the MIT License. See the `LICENSE` file for details (create a LICENSE file).

## Acknowledgements

This research was supported by [Add any relevant funding sources or acknowledgements here].
