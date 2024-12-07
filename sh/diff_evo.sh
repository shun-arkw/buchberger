gb_type="shape" # shape, cauchy
dataset_type="backward" # backward, forward
nvars=4 # 2, 3, 4, 5
field="GF7" # GF7, GF31, QQ, RR
density=1.0  # density controlled 
dataset_size=1000 # 1000
popsize=10
operation="mean" # operation is "mean" or "median"  
updating="immediate" # updating is "immediate" or "deferred"
timeout=1
threshold=1000000



if [[ $(echo "$density != 1.0" | bc -l) -eq 1 ]]; then
    data_name="${gb_type}_n=${nvars}_field=${field}_density=${density}"
else
    data_name="${gb_type}_n=${nvars}_field=${field}"
fi

input_dir=data/DE/${dataset_type}/${gb_type}/${data_name}/dataset

save_dir=result/DE/${dataset_type}/nvars=${nvars}/datasize=${dataset_size}/${operation}
mkdir -p $save_dir


sage src/diff_evo.sage \
    --input_path $input_dir \
    --save_path $save_dir \
    --nvars $nvars \
    --field $field \
    --dataset_size $dataset_size \
    --diff_evo_popsize $popsize \
    --diff_evo_maxiter 1000 \
    --diff_evo_workers 1 \
    --diff_evo_updating $updating \
    --buchberger_timeout $timeout \
    --buchberger_threshold $threshold \
    --upper_bound 100 \
    --lower_bound " -100" \
    --operation $operation > "${save_dir}/run_${data_name}.log"




