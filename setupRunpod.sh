#!/bin/sh

# 1) Setup linux dependencies
su -c 'apt-get update && apt-get install -y sudo'
sudo apt-get install -y less nano htop ncdu nvtop lsof rsync btop jq

# 2) Setup virtual environment
curl -LsSf https://astral.sh/uv/install.sh | sh
. $HOME/.local/bin/env
uv python install 3.11
uv venv
. .venv/bin/activate
uv pip install torch numpy diffusers transformers ipykernel simple-gpu-scheduler # very useful on runpod with multi-GPUs https://pypi.org/project/simple-gpu-scheduler/
python -m ipykernel install --user --name=venv # so it shows up in jupyter notebooks within vscode
# 4) Setup github
git config --global user.name "Phylliida Dev"
git config --global user.email "phylliida.dev@gmail.com"
