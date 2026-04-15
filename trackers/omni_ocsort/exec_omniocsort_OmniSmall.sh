taskset --cpu-list 0 python3 example.py \
    --path_data ../../dataset/OmniSmall/ \
    --input_label_name gt.txt \
    --name_algo omni_ocsort \
    --list_cost_types "giou,euc" \
    --list_weights "0.5,0.5" \
    --threshold 0.3