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

Pre-trained checkpoints are hosted on HuggingFace. You can download using the HuggingFace CLI or `wget`.

**A. Using HuggingFace CLI:**
```bash
pip install huggingface_hub
huggingface-cli login  # Authenticate with your HuggingFace account/token

# Clone the checkpoint repository
huggingface-cli repo clone lerobot/pusht-act-checkpoint
```

**B. Using wget (direct download):**
```bash
wget https://huggingface.co/lerobot/pusht-act-checkpoint/resolve/main/pusht_act.ckpt -O checkpoints/pusht_act.ckpt
wget https://huggingface.co/lerobot/pusht-diffusion-checkpoint/resolve/main/pusht_diffusion.ckpt -O checkpoints/pusht_diffusion.ckpt
```

> **If the checkpoint repository is private you must provide a Hugging Face access token:**  
> ```bash
> export HF_TOKEN=&lt;your_token&gt; ; bash scripts/download_checkpoints.sh
> ```

**Checkpoint Table:**

| Policy    | File Name               | Size     | HuggingFace Repo                          |
|-----------|------------------------|----------|--------------------------------------------|
| ACT       | `pusht_act.ckpt`       | ~80 MB   | `lerobot/pusht-act-checkpoint`             |
| Diffusion | `pusht_diffusion.ckpt` | ~120 MB  | `lerobot/pusht-diffusion-checkpoint`       |

> Place downloaded checkpoints in your `checkpoints/` directory (create if not present).

---

## 4. Running Evaluation

Use the provided evaluation script to run policy evaluation on Push-T.

**Example command:**
```bash
python external/lerobot/examples/evaluate_policy.py \
  --checkpoint checkpoints/pusht_act.ckpt \
  --task push_t \
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
*(Numbers are typical for the ACT baseline; they may vary slightly run-to-run.)*

**Saving results to JSON:**
```bash
python external/lerobot/examples/evaluate_policy.py \
  --checkpoint checkpoints/pusht_act.ckpt \
  --task push_t \
  --episodes 100 \
  --device cuda \
  --output results/pusht_act_eval.json
```

---

## 5. Troubleshooting & Tips

- **CUDA Mismatch:** If you see errors about CUDA versions, ensure your PyTorch and CUDA toolkit versions are compatible.
- **Dataset Not Found:** Make sure `$LEROBOT_DATA_DIR` is set and contains the Push-T dataset. Re-run the dataset download command if necessary.
- **Checkpoint Not Found:** Double-check the checkpoint path and file name.
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