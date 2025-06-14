# CTM-ACT Push-T Integration

# Installation & Environment Setup

## 1. Python Version

This project requires **Python 3.11** or newer.  
Use either a Conda environment (recommended) or a manually created virtual environment with Python 3.11+.  
**Do not mix conda and venv for this project. Use one or the other.**

---

## 2. Recommended: Conda Environment

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
linkedin.com
+1
venturebeat.com
+1
arxiv.org
+2
reduct.store
+2
physicalintelligence.company
+2
youtube.com
+6
github.com
+6
github.com
+6
arxiv.org
+10
github.com
+10
noailabs.medium.com
+10

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
github.com
+2
github.com
+2
x.com
+2
github.com
bdtechtalks.com
+1
venturebeat.com
+1
arxiv.org
+4
huggingface.co
+4
github.com
+4
.

Trajectory Smoothness / Latency: match or better ACT smoother paths 
medium.com
.

Visualizations

Tick-wise latent traces: heatmaps showing neuron activation & sync events per tick.

Attention & focus maps: overlay action decisions per tick sequence.

Adaptive compute stats: histogram of ticks per episode across difficulties.

Trajectory comparisons: plot ACT vs CTM-ACT jointly, highlighting smoother joint/position paths.

LaTeX Tech Report

After surpassing baselines, generate a structured LaTeX report with sections: Abstract, Intro, Related Work (ACT/diffusion/CTM), Method (CTM-ACT integration), Experiments, Results (metrics & ablations), Visualizations, Discussion, Conclusion, Future Work.

Auto-generate report outline + charts (via matplotlib → TikZ or PNG + embed).

Compile to PDF and save as CTM-ACT_PushT_report.pdf.

🔄 Expected Agent Behavior
Generate a modular PyTorch implementation of CTM synapse wrapper that can “plug into” ACT layers.

Provide training loop with tick scheduling, loss functions, evaluation hooks.

Include logging and visualization scripts for internal state/time-tick traces and trajectory metrics.

Write Jupyter notebooks to produce figures.

After metrics beat baseline, produce templated LaTeX .tex file and compile PDF.