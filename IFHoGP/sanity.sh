#!/bin/bash

#========================================================

bash test_all_folds.sh heat sf 5 21
bash test_all_folds.sh heat dmf 5 21
bash test_all_folds.sh heat ifc_ode 5 21
bash test_all_folds.sh heat ifc_gpt 5 21
bash test_all_folds.sh heat hogp 5 21

bash test_all_folds.sh heat sf 10 21
bash test_all_folds.sh heat dmf 10 21
bash test_all_folds.sh heat ifc_ode 10 21
bash test_all_folds.sh heat ifc_gpt 10 21
bash test_all_folds.sh heat hogp 10 21

bash test_all_folds.sh heat sf 20 21
bash test_all_folds.sh heat dmf 20 21
bash test_all_folds.sh heat ifc_ode 20 21
bash test_all_folds.sh heat ifc_gpt 20 21
bash test_all_folds.sh heat hogp 20 21


#========================================================

bash test_all_folds.sh poisson sf 5 21
bash test_all_folds.sh poisson dmf 5 21
bash test_all_folds.sh poisson ifc_ode 5 21
bash test_all_folds.sh poisson ifc_gpt 5 21
bash test_all_folds.sh poisson hogp 5 21

bash test_all_folds.sh poisson sf 10 21
bash test_all_folds.sh poisson dmf 10 21
bash test_all_folds.sh poisson ifc_ode 10 21
bash test_all_folds.sh poisson ifc_gpt 10 21
bash test_all_folds.sh poisson hogp 10 21

bash test_all_folds.sh poisson sf 20 21
bash test_all_folds.sh poisson dmf 20 21
bash test_all_folds.sh poisson ifc_ode 20 21
bash test_all_folds.sh poisson ifc_gpt 20 21
bash test_all_folds.sh poisson hogp 20 21

#========================================================

bash test_all_folds.sh burgers sf 5 21
bash test_all_folds.sh burgers dmf 5 21
bash test_all_folds.sh burgers ifc_ode 5 21
bash test_all_folds.sh burgers ifc_gpt 5 21
bash test_all_folds.sh burgers hogp 5 21

bash test_all_folds.sh burgers sf 10 21
bash test_all_folds.sh burgers dmf 10 21
bash test_all_folds.sh burgers ifc_ode 10 21
bash test_all_folds.sh burgers ifc_gpt 10 21
bash test_all_folds.sh burgers hogp 10 21

bash test_all_folds.sh burgers sf 20 21
bash test_all_folds.sh burgers dmf 20 21
bash test_all_folds.sh burgers ifc_ode 20 21
bash test_all_folds.sh burgers ifc_gpt 20 21
bash test_all_folds.sh burgers hogp 20 21

#========================================================

bash test_all_folds.sh topopt sf 5 21
bash test_all_folds.sh topopt dmf 5 21
bash test_all_folds.sh topopt ifc_ode 5 21
bash test_all_folds.sh topopt ifc_gpt 5 21
bash test_all_folds.sh topopt hogp 5 21

bash test_all_folds.sh topopt sf 10 21
bash test_all_folds.sh topopt dmf 10 21
bash test_all_folds.sh topopt ifc_ode 10 21
bash test_all_folds.sh topopt ifc_gpt 10 21
bash test_all_folds.sh topopt hogp 10 21

bash test_all_folds.sh topopt sf 20 21
bash test_all_folds.sh topopt dmf 20 21
bash test_all_folds.sh topopt ifc_ode 20 21
bash test_all_folds.sh topopt ifc_gpt 20 21
bash test_all_folds.sh topopt hogp 20 21




