# Running Baseline ACT & Diffusion Policies on Push-T

This guide provides step-by-step instructions for evaluating baseline ACT and Diffusion policies on the LeRobot Push-T environment.

---

## 1. Prerequisites

- **Python 3.11** (required)
- **CUDA-enabled GPU** (recommended for evaluation speed)
- **Mujoco** and **Gymnasium** (for simulation)
- **HuggingFace CLI** (for checkpoint download)
- **LeRobot codebase** (with submodules initialized)

**Setup steps:**
1. Clone the repository and initialize submodules:
   ```bash
   git clone https://github.com/lerobot/lerobot.git
   cd lerobot
   git submodule update --init --recursive
   ```
2. Run the environment setup script (if provided):
   ```bash
   bash scripts/setup_env.sh
   ```
   *(Adjust path/script name if different in your repo.)*

---

## 2. Dataset & Environment Setup

- **Environment:** This baseline uses the LeRobot *Push-T* environment, a tabletop manipulation task.
- **Simulation:** Requires [Mujoco](https://mujoco.org/) and [Gymnasium](https://gymnasium.farama.org/) installed and properly licensed.

**Set the data directory environment variable:**
```bash
export LEROBOT_DATA_DIR=$HOME/lerobot_data
```

**Download the Push-T dataset:**
```bash
python -m lerobot.datasets.download --task push_t --output $LEROBOT_DATA_DIR
```
*If this command differs in your setup, please consult the LeRobot dataset documentation.*

---

## 3. Pre-trained Checkpoint Download

Pre-trained checkpoints are hosted on Hugging Face. You may either:
- **A. Download the weights manually via wget (or the provided script),**
- **B. Or pass the Hugging Face repo-id directly to the evaluation script (no download needed).**

**A. Using wget (direct download):**
```bash
# ACT policy (~210MB)
wget https://huggingface.co/pepijn223/act-pusht/resolve/main/model.safetensors -O checkpoints/pusht_act.safetensors

# Diffusion policy (~1GB)
wget https://huggingface.co/lerobot/diffusion_pusht/resolve/main/model.safetensors -O checkpoints/pusht_diffusion.safetensors
```

> **If the checkpoint repository is private you must provide a Hugging Face access token:**  
> ```bash
> export HF_TOKEN=&lt;your_token&gt; ; bash scripts/download_checkpoints.sh
> ```

**B. (Alternative) Pass Hugging Face repo-id directly:**
You may skip downloading and simply provide the repo-id (e.g. `lerobot/diffusion_pusht`) as the `--checkpoint` argument to the evaluation script.  
The script will automatically download and cache the weights using Hugging Face Hub.

**Checkpoint Table:**

| Policy    | File Name                        | Size     | Hugging Face Repo                    |
|-----------|----------------------------------|----------|--------------------------------------|
| ACT       | pusht_act.safetensors            | ~210 MB  | `pepijn223/act-pusht`                |
| Diffusion | pusht_diffusion.safetensors      | ~1 GB    | `lerobot/diffusion_pusht`            |

> Place downloaded checkpoints in your `checkpoints/` directory (create if not present).

---

## 4. Running Evaluation

Use the provided evaluation script to run policy evaluation on Push-T.

**Example command (local file):**
```bash
python scripts/eval_baseline.py \
  --policy act \
  --checkpoint checkpoints/pusht_act.safetensors \
  --episodes 100 \
  --device cuda
```

**Example command (using Hugging Face repo-id, no download needed):**
```bash
python scripts/eval_baseline.py \
  --policy diffusion \
  --checkpoint lerobot/diffusion_pusht \
  --episodes 100 \
  --device cuda
```

**Expected output metrics:**
```
Evaluating: ACT policy on Push-T
Success Rate: 0.83
Mean Reward: 0.67
Episode Length: 52.1
```
*(Numbers are typical for ACT baseline; they may vary slightly run-to-run.)*

**Saving results to JSON:**
```bash
python scripts/eval_baseline.py \
  --policy act \
  --checkpoint checkpoints/pusht_act.safetensors \
  --episodes 100 \
  --device cuda \
  --output results/pusht_act_eval.json
```

---

## 5. Troubleshooting & Tips

- **CUDA Mismatch:** If you see errors about CUDA versions, ensure your PyTorch and CUDA toolkit versions are compatible.
- **Dataset Not Found:** Make sure `$LEROBOT_DATA_DIR` is set and contains the Push-T dataset. Re-run the dataset download command if necessary.
- **Checkpoint Not Found:** Double-check the checkpoint path, file name, or Hugging Face repo-id.
- **Hugging Face Hub Auth:** Some models may require a login or access token.
- **Mujoco License:** Mujoco requires a license; see [Mujoco website](https://mujoco.org/) for details.
- **Environment Variable:** Always `export LEROBOT_DATA_DIR` in the same shell before running scripts.
- **Simulation Fails to Render:** On headless servers, use `--no-render` flag if available, or configure a virtual display (`xvfb`).

---

## 6. (Optional) Re-training the Baseline

To train a baseline policy from scratch:

```bash
python external/lerobot/examples/train_policy.py \
  --task push_t \
  --algo act \
  --output checkpoints/pusht_act_retrained.ckpt \
  --device cuda
```
*(Adjust script/args as needed for your setup.)*

---