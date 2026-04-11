taskset --cpu-list 0 python3 example.py \
    --path_data /home/xins/OmniSORT/dataset/omni_small/data \
    --input_label_name gt.txt \
    --sort_max_age 10 \
    --sort_min_hits 1 \
    --threshold 0.3 \
    --name_algo sort_ori 