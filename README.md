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

# Citation
Please cite the following if you use the data or the model in your work:

1. Geometric Model
```bibtex
@ARTICLE{9932432,
  author={Jiang, Xiajun and Toloubidokhti, Maryam and Bergquist, Jake and Zenger, Brian and Good, Wilson W. and MacLeod, Rob S. and Wang, Linwei},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Improving Generalization by Learning Geometry-Dependent and Physics-Based Reconstruction of Image Sequences}, 
  year={2023},
  volume={42},
  number={2},
  pages={403-415},
  keywords={Geometry;Heart;Image reconstruction;Physics;Training data;Imaging;Torso;Geometric deep learning;inverse problems;physics-based deep learning},
  doi={10.1109/TMI.2022.3218170}}

@inproceedings{jiang2020learning,
  title={Learning geometry-dependent and physics-based inverse image reconstruction},
  author={Jiang, Xiajun and Ghimire, Sandesh and Dhamala, Jwala and Li, Zhiyuan and Gyawali, Prashnna Kumar and Wang, Linwei},
  booktitle={Medical Image Computing and Computer Assisted Intervention--MICCAI 2020: 23rd International Conference, Lima, Peru, October 4--8, 2020, Proceedings, Part VI 23},
  pages={487--496},
  year={2020},
  organization={Springer}
}
```

2. Hybrid SSM
```bibtex
@ARTICLE{10471622,
  author={Jiang, Xiajun and Missel, Ryan and Toloubidokhti, Maryam and Gillette, Karli and Prassl, Anton J. and Plank, Gernot and Milan Horáček, B. and Sapp, John L. and Wang, Linwei},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Hybrid Neural State-Space Modeling for Supervised and Unsupervised Electrocardiographic Imaging}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Electrocardiography;Bayes methods;Heart;Filtering;Data models;Image reconstruction;Physics;Image reconstruction;Bayesian filter;Neural ODE;Graph convolution},
  doi={10.1109/TMI.2024.3377094}}

@inproceedings{jiang2021label,
  title={Label-free physics-informed image sequence reconstruction with disentangled spatial-temporal modeling},
  author={Jiang, Xiajun and Missel, Ryan and Toloubidokhti, Maryam and Li, Zhiyuan and Gharbia, Omar and Sapp, John L and Wang, Linwei},
  booktitle={Medical Image Computing and Computer Assisted Intervention--MICCAI 2021: 24th International Conference, Strasbourg, France, September 27--October 1, 2021, Proceedings, Part VI 24},
  pages={361--371},
  year={2021},
  organization={Springer}
}
```

3. Personalized Neural Surrogate
```bibtex
@inproceedings{
jiang2023sequential,
title={Sequential Latent Variable Models for Few-Shot High-Dimensional Time-Series Forecasting},
author={Xiajun Jiang and Ryan Missel and Zhiyuan Li and Linwei Wang},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=7C9aRX2nBf2}
}

@inproceedings{jiang2022few,
  title={Few-Shot Generation of Personalized Neural Surrogates for Cardiac Simulation via Bayesian Meta-learning},
  author={Jiang, Xiajun and Li, Zhiyuan and Missel, Ryan and Zaman, Md Shakil and Zenger, Brian and Good, Wilson W and MacLeod, Rob S and Sapp, John L and Wang, Linwei},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={46--56},
  year={2022},
  organization={Springer}
}
```
