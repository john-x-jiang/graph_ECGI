#!/bin/bash

CUDA=cu113

#pip install -r requirements.txt

# pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
# pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
# pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
# pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
# pip install torch-geometric

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.10.0+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.10.0+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.10.0+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.10.0+${CUDA}.html
pip install torch-geometric
