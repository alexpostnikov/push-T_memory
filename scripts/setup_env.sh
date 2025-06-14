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
    git clone --depth 1 https://github.com/lerobot/lerobot.git external/lerobot
else
    info "external/lerobot already exists; skipping clone."
fi

# Clone SakanaAI Continuous Thought Machine (CTM) if missing
if [ ! -d external/ctm ]; then
    info "Cloning SakanaAI Continuous Thought Machine into external/ctm..."
    git clone --depth 1 https://github.com/SakanaAI/Continuous-Thought-Machine.git external/ctm
else
    info "external/ctm already exists; skipping clone."
fi

# Install Python requirements
info "Installing Python requirements..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Install LeRobot and CTM in editable mode
info "Installing LeRobot and CTM in editable mode..."
python3 -m pip install -e external/lerobot
python3 -m pip install -e external/ctm

success "Environment setup complete! You can now develop and run experiments."