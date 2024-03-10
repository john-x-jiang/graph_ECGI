#!/bin/bash

CUDA=cu117

pip install -r requirements.txt

pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.1+${CUDA}.html
