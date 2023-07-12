# Probability-Surrogate-Learning
This is a comprehensive machine learning library, specifically tailored for surrogate learning and active learning. Our objective is to provide a robust platform that empowers researchers and developers to seamlessly implement and experiment with these advanced machine learning techniques.

# IFHoGP-main(Scalable High-Order Gaussian Process Regression)
First of all, you need to download our [data](https://drive.google.com/file/d/1ekh_rITLrXvPyThm7DSid8fl8YuyYQVK/view?usp=sharing) for testing and copy them into a folder named pde_data.

You can easily run our code by running the following command:
```commandline
cd IFHoGP-main
ipython test_script.ipy
```

# Scalable-GPRN(Scalable Gaussian Process Regression Networks)

You can easily run our code by running the following command:
```commandline
cd Scalable-GRPN
python sgprn.py
```

# Deep-Coregionalization(Deep Coregionalization for the Emulation of Simulation-Based Spatial-Temporal Fields)
You can easily run our code by running the following command:

```commandline
cd Deep-Coregionalization
python run-baselines.py -domain=Heat -method=dc -rank=5 -fold=1
```

If you need more data(e.g. different domain(Burgers, Poisson, NS), different method, different fold), you can get from IFC(Infinite-Fidelity-Coregionalization) project by running
```commandline
cd Infinite-Fidelity-Coregionalization
python process-data-drc.py
```
Therefore, you can copy the data from DRC-pde-data folder to exp_data_dc in Deep-Coregionalization.

# Multi-Fidelity-High-Order-Gaussian-Processes-for-Physical-Simulation
You can easily run our code by running the following command:
```commandline
cd Multi-Fidelity-High-Order-Gaussian-Processes-for-Physical-Simulation && cd DGP_hd
python test.py
```
# DMFAL(Deep Multi-Fidelity Active Learning of High-Dimensional Outputs)
You can easily run our code by running the following command:
```commandline
cd Active-Learning && cd DMFAL && cd scripts && cd Poisson2
bash run-Poisson2-pdv.sh
```
# BMFAL-BC(Batch Multi-Fidelity Active Learning with Budget Constraints)
You can easily run our code by running the following command:
```commandline
cd Active-Learning && cd BMFAL-BC
bash experiment.sh
```
# Infinite-Fidelity-Coregionalization(Infinite-Fidelity Coregionalization for Physical Simulation)
You can easily run our code by running the following command:
```commandline
cd Infinite-Fidelity-Coregionalization
bash test-ODE.sh/test-GPT.sh Heat 5 500 cuda:0 0 10
```
# Data-Generator
You can generate the solutions of the Heat, Poisson, Burgers, NavierStockURec by running the following command:
```commandline
cd Data-Generator
python generate.py -domain=(Heat, Poisson, Burgers, TopOpt, NavierStockPRec/URec/VRec)
```
More information and instructions are in Data-Generator README.md.