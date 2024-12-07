dataset_type="backward" # backward, forward
gb_type="shape" # shape, cauchy
nvars=4 # 2, 3, 4, 5
field="GF7" # GF7, GF31, QQ, RR
density=1.0  # density controlled 

train_dataset_size=5000 # 1000
test_dataset_size=15000 # 1000

train_timeout=0.5
test_timeout=5
threshold=10000000



if [[ $(echo "$density != 1.0" | bc -l) -eq 1 ]]; then
    data_name="${gb_type}_n=${nvars}_field=${field}_density=${density}"
else
    data_name="${gb_type}_n=${nvars}_field=${field}"
fi

input_dir=data/origin/${dataset_type}/${gb_type}/${data_name}/dataset

save_dir=data/DE/${dataset_type}/${gb_type}/${data_name}
mkdir -p $save_dir


sage src/dataset/build_DE_dataset.sage \
    --input_path $input_dir \
    --save_path $save_dir \
    --config_path "config/${dataset_type}/${data_name}.yaml" \
    --nvars $nvars \
    --field $field \
    --test_dataset_size $test_dataset_size \
    --train_dataset_size $train_dataset_size \
    --buchberger_train_timeout $train_timeout \
    --buchberger_test_timeout $test_timeout \
    --buchberger_threshold $threshold  > "${save_dir}/run_${data_name}.log"




