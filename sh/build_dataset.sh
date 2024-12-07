#!/bin/bash

############################################
dataset_type="backward" # backward, forward
gb_type="shape" # shape, cauchy
num_variables=5 # 2, 3, 4, 5
field="GF7" # GF7, GF31, QQ, RR
density=1.0  # density controlled (refer to the paper)
# density=1.0                          # full density

############################################



if [[ $(echo "$density != 1.0" | bc -l) -eq 1 ]]; then
    config="${gb_type}_n=${num_variables}_field=${field}_density=${density}"
else
    config="${gb_type}_n=${num_variables}_field=${field}"
fi
save_dir="data/origin/${dataset_type}/${gb_type}/${config}"
mkdir -p "$save_dir"

echo "Running configuration: $config with density $density"

sage src/dataset/build_dataset.sage \
    --save_path "$save_dir" \
    --config_path "config/${dataset_type}/${config}.yaml" > "${save_dir}/run_${config}.log" # 2>&1
    
    # --testset_only \  # if you only need testset


echo "Completed: $config"
echo "------------------------"
