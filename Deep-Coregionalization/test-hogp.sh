#!/bin/bash

rank=$1

python run-baselines.py -domain=Heat -method=hogp -rank=$rank -fold=0
python run-baselines.py -domain=Heat -method=hogp -rank=$rank -fold=1
python run-baselines.py -domain=Heat -method=hogp -rank=$rank -fold=2
python run-baselines.py -domain=Heat -method=hogp -rank=$rank -fold=3
python run-baselines.py -domain=Heat -method=hogp -rank=$rank -fold=4

python run-baselines.py -domain=Poisson -method=hogp -rank=$rank -fold=0
python run-baselines.py -domain=Poisson -method=hogp -rank=$rank -fold=1
python run-baselines.py -domain=Poisson -method=hogp -rank=$rank -fold=2
python run-baselines.py -domain=Poisson -method=hogp -rank=$rank -fold=3
python run-baselines.py -domain=Poisson -method=hogp -rank=$rank -fold=4

python run-baselines.py -domain=Burgers -method=hogp -rank=$rank -fold=0
python run-baselines.py -domain=Burgers -method=hogp -rank=$rank -fold=1
python run-baselines.py -domain=Burgers -method=hogp -rank=$rank -fold=2
python run-baselines.py -domain=Burgers -method=hogp -rank=$rank -fold=3
python run-baselines.py -domain=Burgers -method=hogp -rank=$rank -fold=4


