#!/bin/bash
pip install -q torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
echo "PyTorch installed!"

pip install -q -r "install/requirements.txt"

git clone --quiet https://github.com/Jacobe2169/GMatch4py.git
cd GMatch4py
pip install -q .
cd ..

echo "All Libraries installed!"