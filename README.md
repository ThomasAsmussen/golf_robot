# golf_robot
        🤖
       /|\
        |  |
       / \ |
           |OO| ~~~  o  ---->  ⛳

UR10 robot playing minigolf with reinforcement learning.

This repo contains code for training and evaluating RL policies that choose a golf swing (e.g. **impact speed** and **swing angle**) given an observed setup (ball position, hole position). The current training setup is largely *single-step* / *contextual bandit style* (choose one action → get one outcome/reward).  
Training scripts save checkpoints under `models/`.

This repository contains training loops, environment interfaces, utilities, experiment scripts, and infrastructure for running reinforcement learning experiments in both simulation and real-robot settings.

> Status: experimental / research code. Expect rough edges, refactors, and hard-coded paths in HPC scripts.

The primary implementation lives in:

    src/golf_robot/

Everything else (configs, models, notebooks, cluster scripts) supports the workflows defined there.

---

## Policy Evaluation
policy_evaluation contains the policy evaluation plots and mat files

## Repository Structure

.
├── src/golf_robot/        # Main Python package (core implementation)
├── configs/               # Experiment and environment configuration files
├── models/                # Saved checkpoints and trained policies
├── data/                  # Raw and processed data (if applicable)
├── notebooks/             # Jupyter notebooks for analysis/experiments
├── reports/figures/       # Generated plots and figures
├── dockerfiles/           # Docker build files
├── run_*.sh               # Cluster/job submission scripts
├── tests/                 # Unit tests
├── pyproject.toml         # Package definition
├── requirements.txt       # Runtime dependencies
└── README.md

---

## Core Package: src/golf_robot

This directory contains:

- Reinforcement learning training scripts (e.g. SAC / TD3 / DDPG variants)
- Environment interaction logic (simulation and/or real robot)
- Model definitions and update logic
- Utility modules
- Experiment orchestration

All runnable entrypoints are defined inside this package.

You can list available Python modules with:

python - <<'PY'
from pathlib import Path
pkg = Path("src/golf_robot")
for p in sorted(pkg.rglob("*.py")):
    rel = p.relative_to(pkg)
    if not rel.name.startswith("_"):
        print(rel.as_posix())
PY

---

## Installation

Python >= 3.10 is required.

From the repository root:

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

Alternatively:

pip install -r requirements.txt

---

## Running Training

Training entrypoints live under golf_robot.

Run a module as:

python -m golf_robot.<module_name>

Example:

python -m golf_robot.SAC_bandit

To inspect available CLI options:

python -m golf_robot.<module_name> --help

Training artifacts are typically written to:

- models/ for checkpoints
- reports/figures/ for plots
- data/ for rollouts/logs (if used)

Paths may be configurable via YAML files in configs/.

---

## Configuration

Configuration files are located in:

configs/

These may include:

- Reinforcement learning hyperparameters
- Environment parameters
- Logging settings
- Hardware/simulation options

Refer to the specific training module to see how configs are loaded.

For evaluation set "continue_training = True" in the rl config

---

## Real-Time Scheduling (Linux)

For stable robot control loops (e.g. streaming `speedj()` commands at ~125 Hz), the trajectory streaming process may use **real-time scheduling** via `chrt`.

By default, `chrt` requires root privileges. To allow your user to run it without entering a password, add a rule in `sudoers`.

Edit the sudoers file:

```bash
sudo visudo
```

Add the following line at the bottom (replace `<username>` with your Linux username):

```bash
<username> ALL=(root) NOPASSWD: /usr/bin/chrt
```

You can then run a process with FIFO real-time priority, for example:

```bash
sudo chrt -f 80 ./traj_streamer
```

Verify the rule with:

```bash
sudo -l
```
## Recording Experiments (OBS Studio)

OBS Studio can be used to record ball trajectory for training and logging.

Install OBS:

```bash
sudo apt install obs-studio
```
Add the camera using 30 FPS, and the correct recording path in settings -> Output -> Recording -> Recording Path

For specifying which cores to use for OBS Studio, start the program from the terminal using: 

```bash
taskset -c 0-9 obs
```

For cores 0-9. Make sure these are different from the cores used to run traj_streamer (specified in training and evaluation scripts)

## Running on a Cluster

Shell scripts (run_*.sh) are provided for submitting jobs to a scheduler (e.g. BSUB).

Before running:

1. Update any hard-coded paths.
2. Ensure the correct Python environment is activated.
3. Confirm CUDA/modules are loaded if using GPU.

Submit using your scheduler command, for example:

bsub < run_sac.sh

---


## License

See the LICENSE file in the repository root.

# MLOPS
Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
