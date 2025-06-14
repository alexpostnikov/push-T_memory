#!/usr/bin/env bash
set -e

# Helper function to echo with color
info() { echo -e "\033[1;34m[INFO]\033[0m $1"; }
success() { echo -e "\033[1;32m[SUCCESS]\033[0m $1"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $1"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $1"; }

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_ROOT"

# Create external/ directory if missing
if [ ! -d external ]; then
    info "Creating external/ directory..."
    mkdir external
else
    info "external/ directory already exists."
fi

# Clone LeRobot (shallow, default branch) if missing
if [ ! -d external/lerobot ]; then
    info "Cloning LeRobot into external/lerobot..."
    git clone --depth 1 https://github.com/huggingface/lerobot.git external/lerobot
else
    info "external/lerobot already exists; skipping clone."
fi

# Clone SakanaAI Continuous Thought Machine (CTM) if missing
if [ ! -d external/ctm ]; then
    info "Cloning SakanaAI Continuous Thought Machine into external/ctm..."
    git clone --depth 1 https://github.com/SakanaAI/continuous-thought-machines.git external/ctm
else
    info "external/ctm already exists; skipping clone."
fi

# Python virtual environment setup
if [ ! -d venv ]; then
    info "Creating Python virtual environment in ./venv ..."
    python3 -m venv venv
else
    info "Python virtual environment (venv) already exists."
fi

# Activate the virtual environment
# shellcheck disable=SC1091
source venv/bin/activate
info "Activated virtual environment."

# Upgrade pip and install Python requirements
info "Installing Python requirements inside virtual environment..."
pip install --upgrade pip
pip install -r requirements.txt

# Install LeRobot and CTM in editable mode inside venv
info "Installing LeRobot and CTM in editable mode inside venv..."
pip install -e external/lerobot
pip install -e external/ctm

success "Environment setup complete! Virtual environment is active."
echo ""
echo "To deactivate the environment, run: deactivate"
echo "To activate again later, run: source venv/bin/activate"
