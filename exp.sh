#!/bin/bash

# # ST-GCNN Heart to Heart
# python main.py --config demo01 --stage 1
# python main.py --config demo01 --stage 2 --tag qry0,qry1

# # Euclidean (Sandesh et al, IPMI 2019)
# python main.py --config demo02 --stage 1
# python main.py --config demo02 --stage 2 --tag qry0,qry1

# # Meta-model with GRU dynamics (Jiang et al, ICLR 2023)
# python main.py --config demo03 --stage 1
# python main.py --config demo03 --stage 3 --tag qry0,qry1

# # ST-GCNN Inverse problem (Jiang et al, IEEE TMI 2022/MICCAI 2020)
# python main.py --config demo04 --stage 1
# python main.py --config demo04 --stage 2 --tag qry0,qry1

# # Meta-model with Neural ODE dynamics
# python main.py --config demo05 --stage 1
# python main.py --config demo05 --stage 3 --tag qry0,qry1

# Meta-model with GRU dynamics (Jiang et al, MICCAI 2022)
python main.py --config demo06 --stage 1
python main.py --config demo06 --stage 3 --tag qry0,qry1
