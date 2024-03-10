# Running guidance

1. Check your cuda version. I've tested `cu117` and it is able to work.
2. Python version >= `3.8` would be recommended.
3. To install the python packages, change the variable `CUDA` in the script `req_torch_geo.sh`, then run it.
4. To train the model, run the following command (`demo01` for ST-GCNN heart-in-heart-out, `demo02` for Euclidean (Sandesh et al, IPMI 2019), `demo03` for meta-model (Jiang et al, ICLR 2023), `demo04` for ST-GCNN torso-in-heart-out (Jiang et al, IEEE TMI 2022 and MICCAI 2020), `demo05` for meta-model with neural ODE, `demo06` for meta-model with initial condition masks (Jiang et al, MICCAI 2022)):
   ```bash
   python main.py --config demo01 --stage 1
   ```
5. To evaluate the model, run the following command:
   ```bash
   python main.py --config demo01 --stage 2
   ```
6. To perform meta-evaluation on the model, run the following command:
   ```bash
   python main.py --config demo03 --stage 3
   ```
7. To make graphs based on the pre-defined geometry, run the following command:
    ```bash
   python main.py --config demo01 --stage 0
   ```
8. To save evaluation results for visualization, add `--tag <file name>` to the end of evaluation command.
9. TODO list
   1.  Add Hybrid SSM (Jiang et al, IEEE TMI 2024 and MICCAI 2021) to the repo