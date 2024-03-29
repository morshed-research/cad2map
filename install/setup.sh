#!/bin/bash
pip install -q torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
echo "PyTorch installed!"

pip install -q -r "install/requirements.txt"
echo "All Libraries Installed!"