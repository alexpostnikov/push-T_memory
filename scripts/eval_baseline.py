import argparse
import os
import sys
import time
import json
import random
from collections import defaultdict

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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a baseline policy on the environment.",
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        required=True,
        help="Path to the policy file (.pt, .safetensors, or HuggingFace repo-id).",
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Environment name.",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment.",
    )
    return parser.parse_args()


class PolicyWrapper:
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

def main():
    args = parse_args()
    device = args.device or auto_device()
    episodes = args.episodes
    policy_type = args.policy
    checkpoint = args.checkpoint
    output_path = args.output
    seed = args.seed

    print(f"Evaluating {policy_type.upper()} policy on LeRobot-PushT-v0")
    print(f"Checkpoint (local path or Hugging Face repo-id): {checkpoint}")
    print(f"Episodes: {episodes}")
    print(f"Device: {device}")

    if seed is not None:
        set_global_seed(seed)
        print(f"Using global seed: {seed}")

    # Load policy
    try:
        policy = PolicyWrapper(policy_type, checkpoint, device)
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
        success_rate = success_count / episodes,
        mean_reward = total_reward / episodes,
        mean_ep_len = total_steps / episodes,
        mean_latency = total_latency / episodes,
    )

    print("==== Baseline Evaluation Results ====")
    print(f"Success Rate: {agg_metrics['success_rate']:.3f}")
    print(f"Mean Reward: {agg_metrics['mean_reward']:.3f}")
    print(f"Mean Episode Length: {agg_metrics['mean_ep_len']:.2f}")
    print(f"Mean Policy Latency (secs): {agg_metrics['mean_latency'] * 1000:.2f} ms")

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