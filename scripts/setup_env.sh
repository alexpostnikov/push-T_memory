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

# Check for git
if ! command -v git &>/dev/null; then
    warn "git is not installed or not found in PATH. Please install git to use submodules."
else
    info "Initialising / updating git submodules..."
    git submodule update --init --recursive
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

# Install LeRobot and CTM dependencies if present, then install in editable mode
for MODULE in lerobot ctm; do
    if [ -f "external/$MODULE/requirements.txt" ]; then
        info "Installing requirements for $MODULE..."
        pip install -r "external/$MODULE/requirements.txt"
    fi
    info "Installing $MODULE in editable mode..."
    pip install -e "external/$MODULE"
done

success "Environment setup complete! Virtual environment is active."
echo ""
echo "To deactivate the environment, run: deactivate"
echo "To activate again later, run: source venv/bin/activate"
