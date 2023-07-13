#!/bin/bash

python process-data-drc.py --config=configs/heat/exp_sf.py --config.data.fold=1
python process-data-drc.py --config=configs/heat/exp_sf.py --config.data.fold=2
python process-data-drc.py --config=configs/heat/exp_sf.py --config.data.fold=3
python process-data-drc.py --config=configs/heat/exp_sf.py --config.data.fold=4
python process-data-drc.py --config=configs/heat/exp_sf.py --config.data.fold=5

python process-data-drc.py --config=configs/poisson/exp_sf.py --config.data.fold=1
python process-data-drc.py --config=configs/poisson/exp_sf.py --config.data.fold=2
python process-data-drc.py --config=configs/poisson/exp_sf.py --config.data.fold=3
python process-data-drc.py --config=configs/poisson/exp_sf.py --config.data.fold=4
python process-data-drc.py --config=configs/poisson/exp_sf.py --config.data.fold=5

python process-data-drc.py --config=configs/burgers/exp_sf.py --config.data.fold=1
python process-data-drc.py --config=configs/burgers/exp_sf.py --config.data.fold=2
python process-data-drc.py --config=configs/burgers/exp_sf.py --config.data.fold=3
python process-data-drc.py --config=configs/burgers/exp_sf.py --config.data.fold=4
python process-data-drc.py --config=configs/burgers/exp_sf.py --config.data.fold=5

python process-data-drc.py --config=configs/topopt/exp_sf.py --config.data.fold=1
python process-data-drc.py --config=configs/topopt/exp_sf.py --config.data.fold=2
python process-data-drc.py --config=configs/topopt/exp_sf.py --config.data.fold=3
python process-data-drc.py --config=configs/topopt/exp_sf.py --config.data.fold=4
python process-data-drc.py --config=configs/topopt/exp_sf.py --config.data.fold=5
