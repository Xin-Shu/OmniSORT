taskset --cpu-list 0 python3 example.py \
    --path_data ../../dataset/OmniSmall/test \
    --input_label_name gt.txt \
    --det_thresh 0.3 \
    --name_algo ocsort_ori
