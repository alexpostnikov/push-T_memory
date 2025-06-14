import argparse
import os
import sys
import time
import json
import random
import csv
import datetime
from collections import defaultdict

from pathlib import Path

try:
    import torch
except ImportError:
    print("PyTorch is required. Please install torch.")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("NumPy is required. Please install numpy.")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm is required. Please install tqdm.")
    sys.exit(1)

################################################################################
# Helper Functions (Legacy)
################################################################################

def auto_device():
    """Return cuda if available, else cpu."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def set_global_seed(seed):
    """Set random, numpy, and torch seeds (for reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

################################################################################
# Legacy Policy Import (backward compatible)
################################################################################

def import_lerobot_policy(policy_type):
    """
    Dynamically import the correct policy class based on type.
    """
    import importlib
    CANDIDATES = {
        "act": [
            "lerobot.common.policies.act.modeling_act.ACTPolicy",
            "lerobot.models.act.ACTPolicy",
            "lerobot.policies.act_policy.ACTPolicy",
            "lerobot.policy.act.ACTPolicy",
        ],
        "diffusion": [
            "lerobot.common.policies.diffusion.modeling_diffusion.DiffusionPolicy",
            "lerobot.models.diffusion.DiffusionPolicy",
            "lerobot.policies.diffusion_policy.DiffusionPolicy",
            "lerobot.policy.diffusion.DiffusionPolicy",
        ],
    }
    if policy_type not in CANDIDATES:
        raise ValueError(f"Unknown policy type: {policy_type}")
    errors = []
    for dotted in CANDIDATES[policy_type]:
        module_part, cls_name = dotted.rsplit('.', 1)
        try:
            module = importlib.import_module(module_part)
            return getattr(module, cls_name)
        except (ImportError, AttributeError) as e:
            errors.append(str(e))
            continue
    err_msg = (
        f"Could not locate {policy_type} policy class in known module paths.\n"
        "Tried import paths (in order):\n  - " + "\n  - ".join(CANDIDATES[policy_type]) + "\n"
        "Please ensure LeRobot is installed, external/lerobot is up-to-date, and your PYTHONPATH includes the correct modules."
    )
    raise ImportError(err_msg + "\nDetailed import errors:\n" + "\n".join(errors))

class LegacyPolicyWrapper:
    """
    Wraps a legacy LeRobot policy loaded via (policy, checkpoint).
    """
    def __init__(self, policy_type, checkpoint_path, device="cpu"):
        PolicyClass = import_lerobot_policy(policy_type)
        self.device = device
        # if checkpoint_path is local dir or file
        if os.path.isdir(checkpoint_path):
            # Assume HF repo cloned locally; call from_pretrained
            self.policy = PolicyClass.from_pretrained(checkpoint_path, device_map=None).to(device)
        elif os.path.isfile(checkpoint_path):
            if checkpoint_path.endswith(('.ckpt', '.pt', '.pth', '.safetensors')):
                try:
                    # Use load_from_checkpoint if lightning ckpt
                    self.policy = PolicyClass.load_from_checkpoint(checkpoint_path, map_location=device)
                except Exception:
                    # fallback torch load
                    state = torch.load(checkpoint_path, map_location=device)
                    self.policy = PolicyClass(PolicyClass.config_class())
                    if 'state_dict' in state:
                        self.policy.load_state_dict(state['state_dict'], strict=False)
                    else:
                        self.policy.load_state_dict(state, strict=False)
            else:
                # treat as directory
                self.policy = PolicyClass.from_pretrained(checkpoint_path).to(device)
        else:
            # treat as HF repo id
            self.policy = PolicyClass.from_pretrained(checkpoint_path).to(device)
        self.policy.eval()

    def act(self, obs):
        obs_tensor = self._obs_to_tensor(obs)
        with torch.no_grad():
            action = self.policy(obs_tensor)
        if isinstance(action, tuple):
            action = action[0]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if hasattr(action, "ndim") and action.ndim > 1:
            action = action.squeeze(0)
        return action

    def _obs_to_tensor(self, obs):
        if isinstance(obs, dict):
            obs_tensor = {k: torch.from_numpy(np.asarray(v)).float().to(self.device) for k, v in obs.items()}
            obs_tensor = {k: v.unsqueeze(0) if v.ndim == 1 else v for k, v in obs_tensor.items()}
        else:
            obs_tensor = torch.from_numpy(np.asarray(obs)).float().to(self.device)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor

################################################################################
# Generic Policy Wrapper (new, renamed for clarity)
################################################################################

class GenericPolicyWrapper:
    """
    New-style policy wrapper, supports new --policy-path loading.
    """
    def __init__(self, policy_path, device="cpu"):
        self.device = device
        # Try to load a torch.jit (safetensors), torch.load, or HuggingFace repo
        if os.path.isfile(policy_path):
            if policy_path.endswith(".safetensors"):
                print(f"Loading local file (safetensors): {policy_path}")
                self.policy = torch.jit.load(policy_path, map_location=device)
            else:
                print(f"Loading local file: {policy_path}")
                self.policy = torch.load(policy_path, map_location=device)
        else:
            print(f"Loading HuggingFace repo: {policy_path}")
            self.policy = torch.hub.load(policy_path, "policy", source="github", map_location=device)
        self.policy.eval()

    def act(self, obs):
        obs_tensor = self._obs_to_tensor(obs)
        with torch.no_grad():
            action = self.policy(obs_tensor)
        # Assume output is tensor, move to CPU and numpy
        if isinstance(action, tuple):
            action = action[0]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if hasattr(action, "ndim") and action.ndim > 1:
            action = action.squeeze(0)
        return action

    def _obs_to_tensor(self, obs):
        # Accept dict or np.ndarray obs
        if isinstance(obs, dict):
            obs_tensor = {k: torch.from_numpy(np.asarray(v)).float().to(self.device) for k, v in obs.items()}
            obs_tensor = {k: v.unsqueeze(0) if v.ndim == 1 else v for k, v in obs_tensor.items()}
        else:
            obs_tensor = torch.from_numpy(np.asarray(obs)).float().to(self.device)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor

################################################################################
# CLI Argument Parsing (hybrid/compatible)
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a baseline policy on the environment.",
    )
    # New-style arg
    parser.add_argument(
        "--policy-path",
        type=str,
        required=False,
        help="Path to the policy file (.pt, .safetensors, or HuggingFace repo-id).",
    )
    # Legacy args
    parser.add_argument(
        "--policy",
        type=str,
        choices=["act", "diffusion"],
        required=False,
        help="Policy type: act or diffusion (legacy CLI)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="Path to checkpoint file or HF repo id (legacy CLI)",
    )
    # Allow both n-episodes and episodes for compatibility
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        help="Number of episodes to evaluate.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Alias for --n-episodes (legacy CLI).",
    )
    # Device (legacy)
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device to use for inference.",
    )
    # Environment
    parser.add_argument(
        "--env",
        type=str,
        default="LeRobot-PushT-v0",
        help="Environment name (default: LeRobot-PushT-v0).",
    )
    # Other unchanged args
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="results",
        help="Directory to save result logs (default: results)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Log results to Weights & Biases",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="push-t-baselines",
        help="wandb project name (default: push-t-baselines)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="wandb entity (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="If set, write results to this JSON file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    # Backward compatibility validation logic
    # If --policy-path is not provided, require both --policy and --checkpoint
    if args.policy_path is None:
        if args.policy is None or args.checkpoint is None:
            parser.error("Either --policy-path OR both --policy and --checkpoint must be provided.")
    # If --policy-path is provided, ignore policy/checkpoint
    # Episodes: use the first provided among --n-episodes, --episodes, else default 100 for legacy
    if args.n_episodes is not None:
        args.episodes_final = args.n_episodes
    elif args.episodes is not None:
        args.episodes_final = args.episodes
    else:
        args.episodes_final = 100
    # Device: use --device if specified, otherwise auto_device()
    if args.device is None:
        args.device_final = auto_device()
    else:
        args.device_final = args.device
    # Env: default already set
    return args

################################################################################
# ENV IMPORT (assume function is present in user codebase)
################################################################################

def import_lerobot_env():
    """
    Import and initialize the LeRobot environment.
    """
    try:
        import gymnasium as gym
        import lerobot.envs  # register envs
        env = gym.make("LeRobot-PushT-v0")
        return env
    except ImportError as e:
        raise ImportError(
            "LeRobot env not found. Initialise submodule: git submodule update --init --recursive"
        )
    except gym.error.Error as e:
        raise RuntimeError(f"Could not create environment: {e}")

################################################################################
# LOGGING HELPERS (unchanged)
################################################################################

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def append_history(log_dir, row_dict):
    csv_path = os.path.join(log_dir, "history.csv")
    is_new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "policy",
                "checkpoint",
                "success_rate",
                "mean_reward",
                "mean_ep_len",
                "mean_latency",
                "episodes",
            ],
        )
        if is_new:
            writer.writeheader()
        writer.writerow(row_dict)

def update_leaderboard(log_dir, key, sr, metrics_dict):
    leaderboard_path = os.path.join(log_dir, "leaderboard.json")
    if os.path.exists(leaderboard_path):
        with open(leaderboard_path, "r") as f:
            leaderboard = json.load(f)
    else:
        leaderboard = {}
    prev = leaderboard.get(key)
    if not prev or sr > prev.get("success_rate", -float("inf")):
        leaderboard[key] = dict(success_rate=sr, **metrics_dict)
        with open(leaderboard_path, "w") as f:
            json.dump(leaderboard, f, indent=2)
        print(f"New best SR for {key}: {sr:.3f}")
        return True
    return False

################################################################################
# MAIN EVAL LOGIC (hybrid policy loading)
################################################################################

def main():
    args = parse_args()

    # Determine policy loading mode and attributes
    if args.policy_path is not None:
        # New-style loading
        policy_path = args.policy_path
        policy_label = os.path.splitext(os.path.basename(policy_path))[0] if policy_path else ""
        checkpoint_label = policy_path
        policy_loader = lambda: GenericPolicyWrapper(policy_path, args.device_final)
    else:
        # Legacy loading
        policy_type = args.policy
        checkpoint_path = args.checkpoint
        policy_label = policy_type
        checkpoint_label = checkpoint_path
        policy_loader = lambda: LegacyPolicyWrapper(policy_type, checkpoint_path, args.device_final)

    env_name = getattr(args, "env", "LeRobot-PushT-v0")
    episodes = getattr(args, "episodes_final", 100)
    render = getattr(args, "render", False)
    log_dir = getattr(args, "log_dir", "results")
    wandb_flag = getattr(args, "wandb", False)
    wandb_project = getattr(args, "wandb_project", "push-t-baselines")
    wandb_entity = getattr(args, "wandb_entity", None)
    output_path = getattr(args, "output", None)
    seed = getattr(args, "seed", None)
    device = getattr(args, "device_final", auto_device())

    print(f"Evaluating policy on {env_name}")
    print(f"Checkpoint (local path or Hugging Face repo-id): {checkpoint_label}")
    print(f"Episodes: {episodes}")
    print(f"Device: {device}")

    if seed is not None:
        set_global_seed(seed)
        print(f"Using global seed: {seed}")

    # Load policy
    try:
        policy = policy_loader()
    except Exception as e:
        print(str(e))
        sys.exit(1)

    # Load environment
    try:
        env = import_lerobot_env()
    except Exception as e:
        print(str(e))
        sys.exit(1)

    # Set env seed (gymnasium)
    if hasattr(env, 'reset'):
        if seed is not None:
            try:
                env.reset(seed=seed)
            except TypeError:
                env.seed(seed)

    episode_metrics = []
    success_count = 0
    total_reward = 0
    total_steps = 0
    total_latency = 0.0

    for ep in tqdm(range(episodes), desc="Episodes"):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_latency = []
        ep_success = False

        while not done:
            t0 = time.perf_counter()
            action = policy.act(obs)
            latency = time.perf_counter() - t0
            ep_latency.append(latency)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1

            obs = next_obs

        ep_success = info.get("success", False)
        episode_metrics.append({
            "success": bool(ep_success),
            "episode_length": ep_steps,
            "cumulative_reward": float(ep_reward),
            "mean_latency": float(np.mean(ep_latency) if ep_latency else 0.0)
        })
        success_count += int(ep_success)
        total_reward += ep_reward
        total_steps += ep_steps
        total_latency += np.mean(ep_latency) if ep_latency else 0.0

    # Aggregate metrics
    agg_metrics = dict(
        success_rate = success_count / episodes if episodes else 0.0,
        mean_reward = total_reward / episodes if episodes else 0.0,
        mean_ep_len = total_steps / episodes if episodes else 0.0,
        mean_latency = total_latency / episodes if episodes else 0.0,
    )

    print("==== Baseline Evaluation Results ====")
    print(f"Success Rate: {agg_metrics['success_rate']:.3f}")
    print(f"Mean Reward: {agg_metrics['mean_reward']:.3f}")
    print(f"Mean Episode Length: {agg_metrics['mean_ep_len']:.2f}")
    print(f"Mean Policy Latency (secs): {agg_metrics['mean_latency'] * 1000:.2f} ms")

    # --- BEGIN LOGGING AND W&B INTEGRATION ---

    # Ensure the log directory exists
    ensure_dir(log_dir)

    # Prepare row for history.csv
    now = datetime.datetime.now().isoformat(timespec="seconds")
    history_row = dict(
        timestamp=now,
        policy=policy_label,
        checkpoint=checkpoint_label,
        success_rate=agg_metrics["success_rate"],
        mean_reward=agg_metrics["mean_reward"],
        mean_ep_len=agg_metrics["mean_ep_len"],
        mean_latency=agg_metrics["mean_latency"],
        episodes=episodes,
    )
    append_history(log_dir, history_row)

    # Update leaderboard.json
    leaderboard_key = f"{history_row['policy']}|{history_row['checkpoint']}"
    update_leaderboard(log_dir, leaderboard_key, agg_metrics["success_rate"], agg_metrics)

    # Weights & Biases logging (optional)
    if wandb_flag:
        try:
            import wandb
        except ImportError:
            print("WARNING: wandb not installed. Skipping W&B logging.")
        else:
            wandb_kwargs = dict(
                project=wandb_project,
                name=f"{history_row['policy']}_{os.path.basename(history_row['checkpoint'])}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "episodes": episodes,
                    "policy_type": history_row['policy'],
                    "checkpoint": history_row['checkpoint'],
                }
            )
            if wandb_entity:
                wandb_kwargs["entity"] = wandb_entity
            run = wandb.init(**wandb_kwargs)
            wandb.log(agg_metrics)
            wandb.finish()

    # Save output if requested
    if output_path:
        result = {
            "aggregate_metrics": agg_metrics,
            "per_episode_metrics": episode_metrics
        }
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {output_path}")

if __name__ == "__main__":
    main()