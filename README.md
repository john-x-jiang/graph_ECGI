# Running guidance

1. Check your cuda version. I've tested `cu117` and it is able to work.
2. Python version >= `3.8` would be recommended.
3. To install the python packages, change the variable `CUDA` in the script `req_torch_geo.sh`, then run it.
4. Model list:
   1. `demo01`: ST-GCNN heart-in-heart-out
   2. `demo02`: Euclidean (Sandesh et al, IPMI 2019)
   3. `demo03`: meta-model (Jiang et al, ICLR 2023)
   4. `demo04`: ST-GCNN torso-in-heart-out (Jiang et al, IEEE TMI 2022 and MICCAI 2020)
   5. `demo05`: meta-model with neural ODE
   6. `demo06`: meta-model with initial condition masks (Jiang et al, MICCAI 2022)
   7. `demo07`: HybridSSM unsupervised learning (Jiang et al, IEEE TMI 2024 and MICCAI 2021)
   8. `demo08` HybridSSM mixed learning (Jiang et al, IEEE TMI 2024)
5. To train the model, run the following command:
   ```bash
   python main.py --config demo01 --stage 1
   ```
6. To evaluate the model, run the following command:
   ```bash
   python main.py --config demo01 --stage 2
   ```
7. To perform meta-evaluation on the model, run the following command:
   ```bash
   python main.py --config demo03 --stage 3
   ```
8. To make graphs based on the pre-defined geometry, run the following command:
    ```bash
   python main.py --config demo01 --stage 0
   ```
9.  To save evaluation results for visualization, add `--tag <file name>` to the end of evaluation command.