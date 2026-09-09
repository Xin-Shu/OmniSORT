taskset --cpu-list 0 python3 example.py \
    --path_data ../../dataset/OmniSmall/test/ \
    --input_label_name gt.txt \
    --name_algo omni_ocsort_samm_signed_principal_wrap \
    --list_cost_types "giou,omni_euc" \
    --list_weights "0.5,0.5" \
    --threshold 0.3 \
    --speed_correction_method signed_principal_wrap
