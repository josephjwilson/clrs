#!/usr/bin/env bash
# Setup script for CLRS conda environment
# Run disconnect-safe with nohup:
#   nohup bash setup_env.sh > setup.log 2>&1 &
#   tail -f setup.log

set -e

ENV_NAME="clrs"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Creating conda env: $ENV_NAME (Python 3.11)"
conda create -n "$ENV_NAME" python=3.11 -y

echo "==> Activating env"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "==> Installing JAX 0.9.1 with CUDA 12 support"
# CUDA 12 wheels work with CUDA 13.x drivers (backward-compatible)
pip install "jax[cuda12]"

echo "==> Installing tensorflow-cpu (only needed for dataset loading, keeps GPU free)"
pip install "tensorflow-cpu>=2.17.0"

echo "==> Installing tfds-nightly"
pip install "tfds-nightly>=4.9.6.dev202409060044"

echo "==> Installing remaining requirements"
pip install \
  "absl-py>=2.1.0" \
  "attrs>=24.2.0" \
  "chex>=0.1.86" \
  "dm-haiku>=0.0.12" \
  "ml_collections>=0.1.1" \
  "numpy>=1.26.4" \
  "opt-einsum>=3.3.0" \
  "optax>=0.2.3" \
  "six>=1.16.0" \
  "toolz>=0.12.1"

echo "==> Installing clrs in editable mode"
pip install -e "$REPO_DIR"

echo "==> Verifying JAX sees GPUs"
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())"

echo ""
echo "==> Done! Activate with: conda activate $ENV_NAME"
echo ""
echo "==> Creating persistent data directories"
mkdir -p ~/data/CLRS30/checkpoints

echo ""
echo "==> CLUSTER TIP: Run experiments inside tmux to survive SSH disconnect:"
echo "    tmux new -s clrs-run   # reattach with: tmux attach -t clrs-run"
echo ""
echo "    CUDA_VISIBLE_DEVICES=0 \\"
echo "      python -m clrs.examples.run \\"
echo "        --dataset_path ~/data/CLRS30 \\"
echo "        --checkpoint_path ~/data/CLRS30/checkpoints"
echo ""
echo "    - CUDA_VISIBLE_DEVICES=0  uses 1 of 2 L4s (cluster etiquette)"
echo "    - ~/data/CLRS30           persistent, survives reboots (not /tmp)"
