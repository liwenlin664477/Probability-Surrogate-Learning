#!/bin/bash

#python main.py --config=configs/heat/exp_sf.py --workdir=__res_heat__
#python main.py --config=configs/heat/exp_dmf.py --workdir=__res_heat__
#python main.py --config=configs/heat/exp_hogp.py --workdir=__res_heat__
#python main.py --config=configs/heat/exp_ifc_gpt.py --workdir=__res_heat__
#python main.py --config=configs/heat/exp_ifc_ode.py --workdir=__res_heat__

#python main.py --config=configs/poisson/exp_sf.py --workdir=__res_poisson__
#python main.py --config=configs/poisson/exp_dmf.py --workdir=__res_poisson__
#python main.py --config=configs/poisson/exp_hogp.py --workdir=__res_poisson__
#python main.py --config=configs/poisson/exp_ifc_gpt.py --workdir=__res_poisson__
#python main.py --config=configs/poisson/exp_ifc_ode.py --workdir=__res_poisson__

#python main.py --config=configs/topopt/exp_sf.py --workdir=__res_topopt__
#python main.py --config=configs/topopt/exp_dmf.py --workdir=__res_topopt__
#python main.py --config=configs/topopt/exp_hogp.py --workdir=__res_topopt__
#python main.py --config=configs/topopt/exp_ifc_gpt.py --workdir=__res_topopt__
#python main.py --config=configs/topopt/exp_ifc_ode.py --workdir=__res_topopt__

DOMAIN=$1
METHOD=$2
RANK=$3
EPOCHS=${4:-5000}

domain_name=$(echo $DOMAIN | tr '[:upper:]' '[:lower:]')
path_prefix="__res_"
path_suffix="__"
save_path="$path_prefix$domain_name$path_suffix"


for FOLD in {1..5};
do
  if [[ "$METHOD" == "sf" ]];
  then

    python main.py --config=configs/$domain_name/exp_sf.py \
      --workdir=$save_path \
      --config.training.epochs=$EPOCHS \
      --config.model.rank=$RANK \
      --config.data.fold=$FOLD

  elif [[ "$METHOD" == "dmf" ]];
  then

    python main.py --config=configs/$domain_name/exp_dmf.py \
      --workdir=$save_path \
      --config.training.epochs=$EPOCHS \
      --config.model.rank=$RANK \
      --config.data.fold=$FOLD

  elif [[ "$METHOD" == "ifc_ode" ]];
  then

    python main.py --config=configs/$domain_name/exp_ifc_ode.py \
      --workdir=$save_path \
      --config.training.epochs=$EPOCHS \
      --config.model.rank=$RANK \
      --config.data.fold=$FOLD

  elif [[ "$METHOD" == "ifc_gpt" ]];
  then

    python main.py --config=configs/$domain_name/exp_ifc_gpt.py \
      --workdir=$save_path \
      --config.training.epochs=$EPOCHS \
      --config.model.rank=$RANK \
      --config.data.fold=$FOLD

  elif [[ "$METHOD" == "hogp" ]];
  then

    python main.py --config=configs/$domain_name/exp_hogp.py \
      --workdir=$save_path \
      --config.training.epochs=$EPOCHS \
      --config.model.rank=$RANK \
      --config.data.fold=$FOLD

  else
      echo "Error: no such method found.."
      exit 1
  fi
done



