from pathlib import Path
import yaml
import os
import uuid
import sys
from rl_common import *
from simulation.run_sim_rl import run_sim

RENDER = True
ALGORITHM = "bandit"
max_num_discs = 0
EPISODES = 10
actor_type = "hand-tuned"  # "hand-tuned" or "rl"

if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Load configurations
    # ------------------------------------------------------------------
    here    = Path(__file__).resolve().parent
    sim_dir = here / "simulation"
    sys.path.append(str(sim_dir))

    project_root        = here.parents[1]
    mujoco_config_path  = project_root / "configs" / "mujoco_config.yaml"
    rl_config_path      = project_root / "configs" / "rl_config.yaml"

    with open(mujoco_config_path, "r") as f:
        mujoco_cfg = yaml.safe_load(f)

    with open(rl_config_path, "r") as f:
        rl_cfg = yaml.safe_load(f)

    mujoco_cfg["sim"]["render"] = RENDER

    # Temporary XML path
    tmp_name     = f"golf_world_tmp_{os.getpid()}_{uuid.uuid4().hex}.xml"
    tmp_xml_path = project_root / "models" / "mujoco" / tmp_name
    mujoco_cfg["sim"]["xml_path"] = str(tmp_xml_path)

    if actor_type == "rl":
        load_actor_path = project_root / "models" / "rl" / ALGORITHM / f"ddpg_actor_{rl_cfg["training"]["model_name"]}"
        actor, device = load_actor(load_actor_path, rl_cfg)
        print(f"Loaded actor from: {load_actor_path}")
        results = evaluation_policy_short(actor, device, mujoco_cfg, rl_cfg, num_episodes=EPISODES, max_num_discs=max_num_discs, env_step=run_sim, env_type="sim")
    
    elif actor_type == "hand-tuned":
        from hand_designed_agent import hand_tuned_policy
        actor = hand_tuned_policy
        device = torch.device("cpu")
        print("Using hand-designed agent.")

        results = evaluation_policy_hand_tuned(
            actor,
            mujoco_cfg,
            rl_cfg,
            num_episodes=EPISODES,
            max_num_discs=max_num_discs,
            env_step=run_sim,
            env_type="sim",
        )

    else:
        raise ValueError(f"Unknown actor type: {actor_type}")
    
    # Evaluate policy

    print("Evaluation results over {} episodes:".format(EPISODES))
    print(f"  Success Rate:     {results[0]*100} %")
    print(f"  Average Reward:    {results[1]}")
    print(f"  Average Distance to Hole: {results[2]} m")
    # Clean up temporary XML
    try:    
        os.remove(tmp_xml_path)
    except FileNotFoundError:
        pass
