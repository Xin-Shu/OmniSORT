taskset --cpu-list 0 python3 example.py \
    --path_data ../../dataset/OmniSmall/test \
    --input_label_name gt.txt \
    --track_thresh 0.6 \
    --name_algo bytetrack_ori
