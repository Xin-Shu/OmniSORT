#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# lambda_omnieuc follows the paper: 0.0 = full GIoU, 1.0 = full OmniEuc.
case_tag="${1:-omni_small_test_det_nms}"
case "$case_tag" in
    omni_small_test_gt)
        path_data="../../dataset/OmniSmall/test/"
        input_label_name="gt.txt"
        ;;
    omni_small_test_det_nms)
        path_data="../../dataset/OmniSmall/test/"
        input_label_name="det_nms.txt"
        ;;
    jrdb_test_gt)
        path_data="../../dataset/JRDBPano/test/"
        input_label_name="gt.txt"
        ;;
    jrdb_test_det)
        path_data="../../dataset/JRDBPano/test/"
        input_label_name="det_nms.txt"
        ;;
    *)
        echo "Unknown case: $case_tag" >&2
        exit 2
        ;;
esac

for spec in \
    "0.0 1.0 l0_0" \
    "0.1 0.9 l0_1" \
    "0.2 0.8 l0_2" \
    "0.3 0.7 l0_3" \
    "0.4 0.6 l0_4" \
    "0.5 0.5 l0_5" \
    "0.6 0.4 l0_6" \
    "0.7 0.3 l0_7" \
    "0.8 0.2 l0_8" \
    "0.9 0.1 l0_9" \
    "1.0 0.0 l1_0"
do
    read -r lambda_omni_euc lambda_giou lambda_tag <<< "$spec"

    taskset --cpu-list 0 python3 example.py \
        --path_data "$path_data" \
        --input_label_name "$input_label_name" \
        --name_algo "omni_sort_${case_tag}_no_minmax_samm_signed_principal_wrap_lambda_omnieuc_${lambda_tag}" \
        --list_cost_types "giou,omni_euc" \
        --list_weights "${lambda_giou},${lambda_omni_euc}" \
        --threshold 0.3 \
        --speed_correction_method signed_principal_wrap
done
