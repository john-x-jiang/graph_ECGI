#!/bin/bash

# Choose either version for environment setup
# Version 1: using CUDA 11.7
CUDA=cu117

pip install -r requirements.txt

pip install torch==2.0.1

pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.1+${CUDA}.html


# Version 2: using CUDA 12.1
CUDA=cu121

pip install -r requirements.txt

pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121

pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.2.2+${CUDA}.html
