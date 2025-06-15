<old_code>
# Push-T Baseline Setup

## Quick Start

1. Clone the repository.
2. Install dependencies with pip:
   ```
   pip install -r requirements.txt
   ```
   This brings in [LeRobot](https://github.com/huggingface/lerobot) (Push-T) automatically.

> **Note:** The `external/lerobot` submodule is deprecated and will be removed. Installation is now handled through pip—no need to manually clone submodules or set PYTHONPATH.

## Dataset & Environment Setup

- Make sure [Mujoco](https://mujoco.org/) and [Gymnasium](https://gymnasium.farama.org/) are installed and properly licensed.
- Set your LeRobot data directory:
  ```bash
  export LEROBOT_DATA_DIR=$HOME/lerobot_data
  ```
- Download the Push-T dataset:
  ```bash
  python -m lerobot.datasets.download --task push_t --output $LEROBOT_DATA_DIR
  ```

## Pre-trained Checkpoints

- Download pre-trained weights from Hugging Face, or use the Hugging Face repo-id directly with the evaluation script.
- Example (download with wget):
  ```bash
  wget https://huggingface.co/pepijn223/act-pusht/resolve/main/model.safetensors -O checkpoints/pusht_act.safetensors
  wget https://huggingface.co/lerobot/diffusion_pusht/resolve/main/model.safetensors -O checkpoints/pusht_diffusion.safetensors
  ```
- Or use the repo-id directly in the evaluation script (no manual download needed).

## Running Evaluation

```bash
python scripts/eval_baseline.py \
  --policy act \
  --checkpoint checkpoints/pusht_act.safetensors \
  --episodes 100 \
  --device cuda
```

If `checkpoints/pusht_act.safetensors` (or the diffusion checkpoint) exists in the
`checkpoints/` directory, you can omit `--checkpoint` and the script will pick it up
automatically.

Or, using Hugging Face repo-id:

```bash
python scripts/eval_baseline.py \
  --policy diffusion \
  --checkpoint lerobot/diffusion_pusht \
  --episodes 100 \
  --device cuda
```

## Troubleshooting

- Ensure all dependencies are installed.
- No need to clone submodules or set PYTHONPATH for LeRobot anymore.
- For issues, see project README or open an issue.
</old_code>
<new_code>
# Push-T Baseline Setup

## Quick Start

1. Clone the repository.
2. Install dependencies with pip:
   ```
   pip install -r requirements.txt
   ```
   This brings in [LeRobot](https://github.com/huggingface/lerobot) (Push-T) automatically.

> **Note:** The `external/lerobot` submodule is deprecated and will be removed. Installation is now handled through pip—no need to manually clone submodules or set PYTHONPATH.

## Dataset & Environment Setup

- Make sure [Mujoco](https://mujoco.org/) and [Gymnasium](https://gymnasium.farama.org/) are installed and properly licensed.
- Set your LeRobot data directory:
  ```bash
  export LEROBOT_DATA_DIR=$HOME/lerobot_data
  ```
- Download the Push-T dataset:
  ```bash
  python -m lerobot.datasets.download --task push_t --output $LEROBOT_DATA_DIR
  ```

## Pre-trained Checkpoints

- Download pre-trained weights from Hugging Face, or use the Hugging Face repo-id directly with the evaluation script.
- Example (download with wget):
  ```bash
  wget https://huggingface.co/pepijn223/act-pusht/resolve/main/model.safetensors -O checkpoints/pusht_act.safetensors
  wget https://huggingface.co/lerobot/diffusion_pusht/resolve/main/model.safetensors -O checkpoints/pusht_diffusion.safetensors
  ```
- Or use the repo-id directly in the evaluation script (no manual download needed).

## Running Evaluation

```bash
python scripts/eval_baseline.py \
  --policy act \
  --checkpoint checkpoints/pusht_act.safetensors \
  --episodes 100 \
  --device cuda
```

Or, using Hugging Face repo-id:

```bash
python scripts/eval_baseline.py \
  --policy diffusion \
  --checkpoint lerobot/diffusion_pusht \
  --episodes 100 \
  --device cuda
```

## Troubleshooting

- Ensure all dependencies are installed.
- No need to clone submodules or set PYTHONPATH for LeRobot anymore.
- For issues, see project README or open an issue.

</new_code>
