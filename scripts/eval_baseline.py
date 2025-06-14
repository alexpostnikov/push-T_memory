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
        description="Evaluate baseline ACT or Diffusion policies on LeRobot Push-T."
    )
    parser.add_argument(
        "--policy", choices=["act", "diffusion"], required=True,
        help="Policy type: act or diffusion"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help=(
            "Path to the pretrained policy checkpoint (.ckpt, .pth, .safetensors) "
            "or a Hugging Face repo-id (e.g. 'lerobot/diffusion_pusht')."
        )
    )
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Number of evaluation episodes (default: 100)"
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None,
        help="Device to use ('cpu', 'cuda', or auto-detect if not set)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional: Output JSON file to write aggregate and per-episode metrics"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Optional: Global random seed"
    )
    return parser.parse_args()

def auto_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def import_lerobot_policy(policy_type):
    try:
        if policy_type == "act":
            from lerobot.models.act import ACTPolicy
            return ACTPolicy
        elif policy_type == "diffusion":
            from lerobot.models.diffusion import DiffusionPolicy
            return DiffusionPolicy
        else:
            raise RuntimeError(f"Unknown policy type: {policy_type}")
    except ImportError as e:
        msg = (
            f"Could not import lerobot policy class for '{policy_type}'.\n"
            "Please ensure that you have initialised the 'external/lerobot' submodule:\n"
            "    git submodule update --init --recursive\n"
            f"Original error: {e}"
        )
        raise ImportError(msg)

def import_lerobot_env():
    try:
        import gymnasium as gym
        # If lerobot registers envs on import, that's enough
        import lerobot.envs
        env = gym.make("LeRobot-PushT-v0")
        return env
    except ImportError as e:
        msg = (
            f"Could not import lerobot environment.\n"
            "Please ensure that you have initialised the 'external/lerobot' submodule:\n"
            "    git submodule update --init --recursive\n"
            f"Original error: {e}"
        )
        raise ImportError(msg)
    except gym.error.Error as e:
        msg = (
            "Could not create environment 'LeRobot-PushT-v0'. Make sure LeRobot is installed and registered correctly.\n"
            f"Original error: {e}"
        )
        raise RuntimeError(msg)

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class PolicyWrapper:
    def __init__(self, policy_type, checkpoint_path, device):
        # Try imports for safetensors and huggingface_hub, error nicely if missing
        try:
            import safetensors
            import huggingface_hub
        except ImportError as e:
            missing = str(e).split("'")[-2] if "'" in str(e) else str(e)
            print(f"Required package '{missing}' is not installed. Please run: pip install safetensors huggingface_hub")
            sys.exit(1)

        PolicyClass = import_lerobot_policy(policy_type)
        self.device = device

        # Check if checkpoint_path is a local file; otherwise, treat as HF repo-id
        if os.path.exists(checkpoint_path):
            # Local file: support .ckpt/.pth/.safetensors
            ext = os.path.splitext(checkpoint_path)[1].lower()
            try:
                self.policy = PolicyClass()
            except Exception as e:
                raise RuntimeError(f"Failed to instantiate {policy_type} policy: {e}")
            if ext == ".safetensors":
                try:
                    from safetensors.torch import load_file
                except ImportError:
                    print("safetensors is required to load .safetensors checkpoints. Install with: pip install safetensors")
                    sys.exit(1)
                state_dict = load_file(checkpoint_path, device="cpu")
            else:
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
            # Remove 'module.' prefix if present (for DDP models)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            try:
                self.policy.load_state_dict(state_dict, strict=False)
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint: {e}")
            self.policy.eval()
            self.policy.to(self.device)
            print(f"✓ Loaded checkpoint from local file: {checkpoint_path}")
        else:
            # Assume Hugging Face Hub repo-id
            try:
                self.policy = PolicyClass.from_pretrained(checkpoint_path, device_map="cpu")
                self.policy.eval()
                self.policy.to(self.device)
                print(f"✓ Loaded checkpoint from Hugging Face Hub repo: {checkpoint_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint from Hugging Face Hub repo-id '{checkpoint_path}': {e}")

    def act(self, obs):
        # Handles single-observation input, returns action (numpy array)
        obs_tensor = self._obs_to_tensor(obs)
        with torch.no_grad():
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    action = self.policy(obs_tensor)
            else:
                action = self.policy(obs_tensor)
        # Assume output is tensor (batch or single), move to CPU and numpy
        if isinstance(action, tuple):
            action = action[0]  # In case model returns (action, info)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if action.ndim > 1:
            action = action.squeeze(0)
        return action

    def _obs_to_tensor(self, obs):
        # Accept dict or np.ndarray obs
        if isinstance(obs, dict):
            obs_tensor = {k: torch.from_numpy(np.asarray(v)).float().to(self.device) for k, v in obs.items()}
        else:
            obs_tensor = torch.from_numpy(np.asarray(obs)).float().to(self.device)
        # Unsqueeze if needed (policy expects batch dim)
        if isinstance(obs_tensor, dict):
            obs_tensor = {k: v.unsqueeze(0) if v.ndim == 1 else v for k, v in obs_tensor.items()}
        else:
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor

    def act(self, obs):
        # Handles single-observation input, returns action (numpy array)
        obs_tensor = self._obs_to_tensor(obs)
        with torch.no_grad():
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    action = self.policy(obs_tensor)
            else:
                action = self.policy(obs_tensor)
        # Assume output is tensor (batch or single), move to CPU and numpy
        if isinstance(action, tuple):
            action = action[0]  # In case model returns (action, info)
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if action.ndim > 1:
            action = action.squeeze(0)
        return action

    def _obs_to_tensor(self, obs):
        # Accept dict or np.ndarray obs
        if isinstance(obs, dict):
            obs_tensor = {k: torch.from_numpy(np.asarray(v)).float().to(self.device) for k, v in obs.items()}
        else:
            obs_tensor = torch.from_numpy(np.asarray(obs)).float().to(self.device)
        # Unsqueeze if needed (policy expects batch dim)
        if isinstance(obs_tensor, dict):
            obs_tensor = {k: v.unsqueeze(0) if v.ndim == 1 else v for k, v in obs_tensor.items()}
        else:
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