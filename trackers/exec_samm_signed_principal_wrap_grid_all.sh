#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# New result names preserve the previous unsigned-modulo experiment batch.
for case_tag in \
    omni_small_test_gt \
    omni_small_test_det_nms \
    jrdb_test_gt \
    jrdb_test_det
do
    ./omni_sort/exec_omnisort_grid.sh "$case_tag"
    ./omni_ocsort/exec_omniocsort_grid.sh "$case_tag"
done
