# CTM-ACT Push-T Integration

---

## 🗺️ Overall Project Plan

This project aims to enhance the Push‑T policy by integrating Continuous Thought Machine (CTM) mechanisms into DeepMind’s ACT (Adaptive Controller Transformer) baseline. The goal is to surpass both ACT and diffusion policy metrics in success rate, trajectory smoothness, and inference speed.

### ✅ What Has Been Done

- **Repository Setup:**  
  - Structured codebase with `src/`, `scripts/`, `notebooks/`, and `report/` directories.
  - Added installation/setup documentation and helper scripts.
- **Baselines Ready:**  
  - ACT and diffusion policy baselines cloned, pre-trained weights loaded, and evaluation scripts set up.
- **External CTM Modules:**  
  - SakanaAI’s CTM repo cloned and basic API/tick/sync modules installed.
- **Initial Integration Plan:**  
  - Outlined approach for wrapping ACT transformer layers with CTM-style modules.
  - Defined evaluation metrics and reporting pipeline.
- **LaTeX Report Template:**  
  - Created initial LaTeX structure for experiment reporting.

### 🔜 What Is Next

- **CTM-Style Integration:**  
  - Implement PyTorch CTM synapse wrappers for ACT transformer layers (internal ticks, neuron history, and sync gating).
  - Replace/augment selected feed-forward and attention layers with tick-synchronous modules.
- **Training Pipeline:**  
  - Fine-tune CTM-ACT model on Push‑T using curriculum tick ramp-up and adaptive loss.
- **Evaluation & Metrics:**  
  - Benchmark against ACT and diffusion policies on success rate, overlap, and trajectory smoothness.
- **Visualization & Logging:**  
  - Develop scripts/notebooks for tick-wise neuron/activity traces, attention maps, and adaptive compute stats.
  - Compare trajectories and plot performance improvements.
- **Reporting:**  
  - Auto-generate updated LaTeX report (metrics, ablations, visualizations).
  - Compile and release PDF of results.

*This section will be updated as milestones are completed. See below for full technical and usage details.*

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo
```

Initialise submodules (required for baseline and CTM modules):

```bash
git submodule update --init --recursive
```

Alternatively, running the setup script below will also initialise submodules automatically.
```bash
conda create -n pushT_ctm python=3.11
conda activate pushT_ctm
pip install -r requirements.txt
```

---

## 3. Alternative: Python 3.11+ venv

If you do **not** use conda, ensure your system Python is 3.11+:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 4. LaTeX Report Compilation

To generate PDF reports, you need a working LaTeX installation with `latexmk` available **on your system path**.  
`latexmk` is **not** a Python package and must be installed via your OS's package manager (e.g., apt, brew, MiKTeX), or you can use Overleaf for online compilation.

---

## Quickstart

Before running any code or notebooks, set up your development environment as described above.  
If you wish to use the helper script, ensure you are already in your Conda or venv environment:

```bash
./scripts/setup_env.sh
```

This script will create external dependencies, install Python requirements, and set up all necessary packages for development and experimentation.

---

## 🚀 Running Baselines

You can evaluate baseline ACT and diffusion policies on the Push‑T environment using pre-trained checkpoints.  
**Follow these steps to run the baselines:**

1. **Install Dependencies**  
   Ensure your environment is set up as described in the [Installation](#installation) and [Quickstart](#quickstart) sections above.

2. **Download Pretrained Checkpoints**  
   - Pretrained weights for ACT and diffusion models can be downloaded via the provided script, or manually using wget, or you may skip downloading and pass a Hugging Face repo-id directly to the evaluation script.
   - To download with the helper script:
     ```bash
     ./scripts/download_checkpoints.sh
     ```
   - Or manually:
     ```bash
     # ACT policy (~210MB)
     wget https://huggingface.co/pepijn223/act-pusht/resolve/main/model.safetensors -O checkpoints/pusht_act.safetensors

     # Diffusion policy (~1GB)
     wget https://huggingface.co/lerobot/diffusion_pusht/resolve/main/model.safetensors -O checkpoints/pusht_diffusion.safetensors
     ```
   - **If the checkpoint repository is private you must provide a Hugging Face access token:**  
     ```bash
     export HF_TOKEN=&lt;your_token&gt; ; bash scripts/download_checkpoints.sh
     ```
   - **Alternatively:** You may skip downloading and simply provide the repo-id (e.g. `lerobot/diffusion_pusht`) as the `--checkpoint` argument to the evaluation script.

3. **Run Baseline Evaluation**  
   - To evaluate the ACT or Diffusion baselines, use the provided evaluation script:
     ```bash
     python scripts/eval_baseline.py --policy act --checkpoint checkpoints/pusht_act.safetensors --episodes 100 --device cuda
     ```
     or, for Diffusion:
     ```bash
     python scripts/eval_baseline.py --policy diffusion --checkpoint checkpoints/pusht_diffusion.safetensors --episodes 100 --device cuda
     ```
   - **(Alternative: Hugging Face repo-id)**  
     You can also run, without manual download:
     ```bash
     python scripts/eval_baseline.py --policy diffusion --checkpoint lerobot/diffusion_pusht --episodes 100 --device cuda
     ```
   - For further options and details, see the script help:
     ```bash
     python scripts/eval_baseline.py --help
     ```

4. **View Results**  
   - Evaluation metrics and logs will be printed to the console, and if `--output` is given, saved to a JSON file.
   - For detailed explanations, see [docs/baselines_push_t.md](docs/baselines_push_t.md).

---

- **Further Details:** See [docs/baselines_push_t.md](docs/baselines_push_t.md) for advanced usage, troubleshooting, and baseline results.

#### Troubleshooting

> **If you receive an ImportError for `lerobot` or the environment:**  
> Make sure you have initialised the `external/lerobot` submodule:
>
> ```bash
> git submodule update --init --recursive
> ```
- If you encounter a "Weights only load failed" error when running the evaluation script under PyTorch 2.6, update to the latest code. The script now sets `weights_only=False` when loading checkpoints.

## Repository Structure

- `external/` – External repositories (added as submodules).
- `src/` – Project Python source code (`ctm_act` pip package).
- `scripts/` – Training, evaluation, and helper scripts.
- `notebooks/` – Jupyter notebooks for analysis and visualization.
- `report/` – LaTeX sources and compiled PDF for experiments and results.

See individual directories for more details as the project develops.

GOAL: Enhance the baseline Push‑T policy built on DeepMind’s ACT (Adaptive Controller Transformer) by integrating Continuous Thought Machine (CTM) mechanisms—specifically neuron-level internal state (“ticks”) and synchronization—to exceed ACT and diffusion policy metrics in metrics like success rate, trajectory smoothness, and inference speed.

 System Overview
Base Architecture

ACT policy: transformer-based autoregressive action predictor trained on Push‑T from LeRobot 
Diffusion policy baseline: shown to slightly outperform ACT in success metrics 

CTM Integration
From SakanaAI's CTM: add an internal temporal axis (ticks) plus per‑neuron short-term history weights and synchronization gating 

Mechanisms: each neuron holds internal activation history, computes across ticks, and synchronizes to form latent for downstream action decoding.

Prompt Steps
Data & Tools Setup

Clone LeRobot repo, load pretrained ACT Push‑T policy, training scripts, and evaluation pipeline.

Clone Sakana1’s CTM repo (https://github.com/SakanaAI/continuous-thought-machines), install CTM API and tick/sync modules. 

Model Integration

Wrap ACT’s transformer layers with CTM-style synaptic modules:

Introduce fixed T internal ticks per step.

Replace feed-forward and attention layers with tick-synchronous U-Net or synapse modules.

Add a gating mechanism where neuron groups fire based on synchronization threshold.

Training Protocol

Start from ACT checkpoint; fine-tune with CTM modifications on Push‑T dataset.

Train with curriculum: ramp-up ticks from T=1→T_max (e.g. 1→5).

Loss: standard imitation + optional tick-adaptive penalty to encourage early stopping on easy tasks (adaptive compute).

Evaluation Metrics

Success Rate: exceed diffusion and ACT on Push‑T environment.

Overlap Ratio: better max overlap vs target (baseline ~0.95) 

Trajectory Smoothness / Latency: match or better ACT smoother paths 

Visualizations

Tick-wise latent traces: heatmaps showing neuron activation & sync events per tick.

Attention & focus maps: overlay action decisions per tick sequence.

Adaptive compute stats: histogram of ticks per episode across difficulties.

Trajectory comparisons: plot ACT vs CTM-ACT jointly, highlighting smoother joint/position paths.

LaTeX Tech Report

After surpassing baselines, generate a structured LaTeX report with sections: Abstract, Intro, Related Work (ACT/diffusion/CTM), Method (CTM-ACT integration), Experiments, Results (metrics & ablations), Visualizations, Discussion, Conclusion, Future Work.


🔄 Expected Agent Behavior
Generate a modular PyTorch implementation of CTM synapse wrapper that can “plug into” ACT layers.

Provide training loop with tick scheduling, loss functions, evaluation hooks.

Include logging and visualization scripts for internal state/time-tick traces and trajectory metrics.

Write Jupyter notebooks to produce figures.

After metrics beat baseline, produce templated LaTeX .tex file and compile PDF.