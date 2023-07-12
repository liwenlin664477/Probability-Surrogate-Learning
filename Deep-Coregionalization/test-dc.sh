#!/bin/bash

python run-baselines.py -domain=Heat -method=dc -rank=5 -fold=1
python run-baselines.py -domain=Heat -method=dc -rank=5 -fold=2
python run-baselines.py -domain=Heat -method=dc -rank=5 -fold=3
python run-baselines.py -domain=Heat -method=dc -rank=5 -fold=4
python run-baselines.py -domain=Heat -method=dc -rank=5 -fold=5

python run-baselines.py -domain=Poisson -method=dc -rank=5 -fold=1
python run-baselines.py -domain=Poisson -method=dc -rank=5 -fold=2
python run-baselines.py -domain=Poisson -method=dc -rank=5 -fold=3
python run-baselines.py -domain=Poisson -method=dc -rank=5 -fold=4
python run-baselines.py -domain=Poisson -method=dc -rank=5 -fold=5

python run-baselines.py -domain=Burgers -method=dc -rank=5 -fold=1
python run-baselines.py -domain=Burgers -method=dc -rank=5 -fold=2
python run-baselines.py -domain=Burgers -method=dc -rank=5 -fold=3
python run-baselines.py -domain=Burgers -method=dc -rank=5 -fold=4
python run-baselines.py -domain=Burgers -method=dc -rank=5 -fold=5

python run-baselines.py -domain=TopOpt -method=dc -rank=5 -fold=1
python run-baselines.py -domain=TopOpt -method=dc -rank=5 -fold=2
python run-baselines.py -domain=TopOpt -method=dc -rank=5 -fold=3
python run-baselines.py -domain=TopOpt -method=dc -rank=5 -fold=4
python run-baselines.py -domain=TopOpt -method=dc -rank=5 -fold=5

python run-baselines.py -domain=Heat -method=dc -rank=10 -fold=1
python run-baselines.py -domain=Heat -method=dc -rank=10 -fold=2
python run-baselines.py -domain=Heat -method=dc -rank=10 -fold=3
python run-baselines.py -domain=Heat -method=dc -rank=10 -fold=4
python run-baselines.py -domain=Heat -method=dc -rank=10 -fold=5

python run-baselines.py -domain=Poisson -method=dc -rank=10 -fold=1
python run-baselines.py -domain=Poisson -method=dc -rank=10 -fold=2
python run-baselines.py -domain=Poisson -method=dc -rank=10 -fold=3
python run-baselines.py -domain=Poisson -method=dc -rank=10 -fold=4
python run-baselines.py -domain=Poisson -method=dc -rank=10 -fold=5

python run-baselines.py -domain=Burgers -method=dc -rank=10 -fold=1
python run-baselines.py -domain=Burgers -method=dc -rank=10 -fold=2
python run-baselines.py -domain=Burgers -method=dc -rank=10 -fold=3
python run-baselines.py -domain=Burgers -method=dc -rank=10 -fold=4
python run-baselines.py -domain=Burgers -method=dc -rank=10 -fold=5

python run-baselines.py -domain=TopOpt -method=dc -rank=10 -fold=1
python run-baselines.py -domain=TopOpt -method=dc -rank=10 -fold=2
python run-baselines.py -domain=TopOpt -method=dc -rank=10 -fold=3
python run-baselines.py -domain=TopOpt -method=dc -rank=10 -fold=4
python run-baselines.py -domain=TopOpt -method=dc -rank=10 -fold=5

python run-baselines.py -domain=Heat -method=dc -rank=20 -fold=1
python run-baselines.py -domain=Heat -method=dc -rank=20 -fold=2
python run-baselines.py -domain=Heat -method=dc -rank=20 -fold=3
python run-baselines.py -domain=Heat -method=dc -rank=20 -fold=4
python run-baselines.py -domain=Heat -method=dc -rank=20 -fold=5

python run-baselines.py -domain=Poisson -method=dc -rank=20 -fold=1
python run-baselines.py -domain=Poisson -method=dc -rank=20 -fold=2
python run-baselines.py -domain=Poisson -method=dc -rank=20 -fold=3
python run-baselines.py -domain=Poisson -method=dc -rank=20 -fold=4
python run-baselines.py -domain=Poisson -method=dc -rank=20 -fold=5

python run-baselines.py -domain=Burgers -method=dc -rank=20 -fold=1
python run-baselines.py -domain=Burgers -method=dc -rank=20 -fold=2
python run-baselines.py -domain=Burgers -method=dc -rank=20 -fold=3
python run-baselines.py -domain=Burgers -method=dc -rank=20 -fold=4
python run-baselines.py -domain=Burgers -method=dc -rank=20 -fold=5

python run-baselines.py -domain=TopOpt -method=dc -rank=20 -fold=1
python run-baselines.py -domain=TopOpt -method=dc -rank=20 -fold=2
python run-baselines.py -domain=TopOpt -method=dc -rank=20 -fold=3
python run-baselines.py -domain=TopOpt -method=dc -rank=20 -fold=4
python run-baselines.py -domain=TopOpt -method=dc -rank=20 -fold=5

