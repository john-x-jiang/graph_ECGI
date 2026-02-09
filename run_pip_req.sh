#!/bin/bash

# This repo relies on the compatibility of pytorch geometric.
# Check the installation guide of pytorch geometric here: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
# E.g. using CUDA 12.9
CUDA=cu129

pip install -r requirements.txt

pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/${CUDA}

pip install torch-geometric

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+${CUDA}.html


