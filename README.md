# Probability-Surrogate-Learning
The Probability-Surrogate-Learning (PSL) library is a comprehensive machine learning toolkit designed to improve predictions and enhance accuracy in physical models. It encompasses various methodologies and techniques that make it a versatile choice for different machine learning tasks, particularly in the physical sciences domain.

You can directly install our library by running:
```commandline
pip install Probability-Surrogate-Learning==1.1
```

---
## Active Learning Techniques
This component allows for interactive and adaptive learning, where the model selectively acquires the most beneficial data to learn from, based on its current understanding.
### [Deep Multi-fidelity Active Learning (DMFAL)](https://arxiv.org/abs/2012.00901)

To run the code:
```commandline
cd Active-Learning/Deep-Multi-fidelity-Active-Learning
bash run-Poisson2-pdv.sh
```
Please cite our work if you would like to use the code
```commandline
@misc{li2021deep,
      title={Deep Multi-Fidelity Active Learning of High-dimensional Outputs}, 
      author={Shibo Li and Zheng Wang and Robert M. Kirby and Shandian Zhe},
      year={2021},
      eprint={2012.00901},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
### [Batch Multifidelity Active Learning with Budget Constraints (BMFAL-BC)](https://arxiv.org/abs/2106.09884)
To run the code:
```commandline
cd Active-Learning/Batch-Multifidelity-Active-Learning-with-Budget-Constraints
bash experiment.sh
```
Please cite our work if you would like to use the code
```commandline
@article{li2021multi,
  title={Batch Multi-Fidelity Bayesian Optimization with Deep Auto-Regressive Networks},
  author={Li, Shibo and Kirby, Robert and Zhe, Shandian},
  journal={Advances in Neural Information Processing Systems},
  year={2021}
}
```
---

## Single-Fidelity Techniques
The Single Fidelity module is designed for traditional machine learning tasks where all data is assumed to have uniform fidelity.
### [Scalable GP Regression Network (SGPRN)](https://arxiv.org/abs/2003.11489)

To run the code:
```commandline
cd Single-Fidelity/Scalable-GP-Regression-Network
python sgprn.py
```
Please cite our work if you would like to use the code
```commandline
@inproceedings{ijcai2020-340,
  title     = {Scalable Gaussian Process Regression Networks},
  author    = {Li, Shibo and Xing, Wei and Kirby, Robert M. and Zhe, Shandian},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  editor    = {Christian Bessiere},	
  pages     = {2456--2462},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/340},
  url       = {https://doi.org/10.24963/ijcai.2020/340},
}
```
### [High-Order Gaussian Process (HOGP)](https://arxiv.org/abs/2003.11489)
This project represents one of the high-order version of Scalable-GPRN, exhibiting certain dependencies within the Scalable-GPRN framework. We provide a visualization notebook as a reference for users:
```commandline
cd Single-Fidelity/High-Order-Gaussian-Process
jupyter notebook Dev-HoGP-v1.ipynb
```
Please cite the paper if you find this code useful.
```commandline
@inproceedings{ijcai2020-340,
  title     = {Scalable Gaussian Process Regression Networks},
  author    = {Li, Shibo and Xing, Wei and Kirby, Robert M. and Zhe, Shandian},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  editor    = {Christian Bessiere},	
  pages     = {2456--2462},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/340},
  url       = {https://doi.org/10.24963/ijcai.2020/340},
}
```
---

## Discrete-Multi-Fidelity 
This module is designed to handle tasks where different levels of accuracy (and computational cost) are available. By combining high- and low-fidelity data, the model can make more accurate predictions with less computational effort.
### [Deep Residual Coregionalization (DRC)](https://arxiv.org/abs/1910.07577)
To run the code:

```commandline
cd Multi-Fidelity/Deep-Residual-Coregionalization
python run-baselines.py -domain=Heat -method=dc -rank=5 -fold=1
```

If you need more data(e.g. different domain(Burgers, Poisson, NS), different method, different fold), you can get from IFC(Infinite-Fidelity-Coregionalization) project by running
```commandline
cd Infinite-or-Continuous-Fidelity/Infinite-Fidelity-Coregionalization
python process-data-drc.py
```
Therefore, you can copy the data from DRC-pde-data folder to exp_data_dc in Deep-Coregionalization.

Please cite the paper if you find this code useful.
```commandline
@article{
XING2021109984, 
title = {Deep coregionalization for the emulation of simulation-based spatial-temporal fields}, 
journal = {Journal of Computational Physics}, 
volume = {428}, 
pages = {109984}, 
year = {2021}, 
issn = {0021-9991}, 
doi = {https://doi.org/10.1016/j.jcp.2020.109984}, 
url = {https://www.sciencedirect.com/science/article/pii/S0021999120307580}, 
author = {Wei W. Xing and Robert M. Kirby and Shandian Zhe}, 
keywords = {Surrogate model, Gaussian process, Emulation, Spatial-temporal field, Multifidelity model}}
```

### [Multi-Fidelity HOGP (MF-HOGP)](https://arxiv.org/abs/2006.04972)
To run the code:

```commandline
cd Multi-Fidelity/Multi-Fidelity-HOGP/DGP_hd
python test.py
```
If you find this useful, or if you use it in your work, please cite:
```
  @article{wang2020multi,
    title={Multi-Fidelity High-Order Gaussian Processes for Physical Simulation},
    author={Wang, Zheng and Xing, Wei and Kirby, Robert and Zhe, Shandian},
    journal={arXiv preprint arXiv:2006.04972}, 
    year={2020}
  }
```

### [Deep Multi-Fidelity (DMF)](http://proceedings.mlr.press/v89/zhe19a/zhe19a.pdf)
To run the code:
```commandline
cd Multi-Fidelity/Deep-Multi-Fidelity
python test_script.py
```
Please cite the paper if you find this code useful.
```commandline
TBD
```
---


## Infinite/Continuous-Fidelity 
This component extends the multi-fidelity concept to situations where fidelity can be any real number, not just discrete levels. This can be useful in situations where there is a continuous scale of accuracy (and associated computational cost).
### [Infinite-Fidelity-Coregionalization(IFC)](https://openreview.net/forum?id=dUYLikScE-)
To run the code:
```commandline
cd Infinite-Or-Continuous-Fidelity/Infinite-Fidelity-Coregionalization
bash test-ODE.sh/test-GPT.sh Heat 5 500 cuda:0 0 10
```
Please cite our paper if you find it helpful :)
```commandline
@inproceedings{
li2022infinitefidelity,
title={Infinite-Fidelity Coregionalization  for Physical Simulation},
author={Shibo Li and Zheng Wang and Robert Kirby and Shandian Zhe},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=dUYLikScE-}
}
```
### [Infinite-Fidelity HOGP (IFHOGP)](http://proceedings.mlr.press/v89/zhe19a/zhe19a.pdf)
First of all, you need to download our [data](https://drive.google.com/file/d/1ekh_rITLrXvPyThm7DSid8fl8YuyYQVK/view?usp=sharing) for testing and copy them into a folder named pde_data.

To run the code:
```commandline
cd Infinite-Or-Continuous-Fidelity/Infinite-fidelity-HOGP
python test_script.py
```
Please cite our paper if you find it helpful :)
```commandline
TBD
```
---

## Data-Generator
You can generate the solutions of the Heat, Poisson, Burgers, NavierStockURec by running the following command:
```commandline
cd Data-Generator
python generate.py -domain=(Heat, Poisson, Burgers, TopOpt, NavierStockPRec/URec/VRec)
```
More information and instructions are in Data-Generator README.md.

---

## Results Visualization
We have also made the result or loss visualization in all the projects. You are welcome to play with them by running jupyter notebook.

---

## License
Probability-Surrogate-Learning is released under the MIT License, please refer the LICENSE for details.

---

## Getting Involved
Feel free to submit Github issues or pull requests. Welcome to contribute to our library!

To contact us, never hestitate to send an email to u1327012@umail.utah.edu or liwen0160@gmail.com
