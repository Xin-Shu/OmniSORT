import os
import sys
import cv2
import datetime
import glob
import shutil
import time
import numpy as np

sys.path.append('../utils/')
import util
from sort import Sort
import argparse


def sort_ori_process(
    ip_input_label, ip_output_label,
    st_frame_num, ed_frame_num, frame_size,
    sort_max_age, sort_min_hits, threshold, cost='iou'
):
    sort = Sort(max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=threshold,
                asso_func=cost, img_size=frame_size)
    f_output_label = open(ip_output_label, 'w')

    max_id = -1
    tracker_runtime_s = 0.0
    dict_input_label = util.load_input_label(ip_input_label, frame_size=frame_size)
    for frame_num in range(st_frame_num, ed_frame_num + 1):
        list_bbox = dict_input_label.get(frame_num, [])
        # load_input_label returns fractional [0,1] coords; SORT expects pixels.
        list_bbox_int = util.box_frac_to_box_int(list_bbox, frame_size)
        if len(list_bbox_int) == 0:
            boxes = np.empty((0, 5))
        else:
            boxes = np.array(list_bbox_int)

        tracker_time_start = time.perf_counter()
        res = sort.update(boxes)
        tracker_runtime_s += time.perf_counter() - tracker_time_start
        boxes_track = res[:, :-1]
        boxes_ids = res[:, -1].astype(int)
        max_id = max(max_id, (max(boxes_ids) if len(boxes_ids) > 0 else -1))

        for box_track_int, id_ in zip(boxes_track, boxes_ids):
            x1_int, y1_int, x2_int, y2_int = box_track_int
            w_int, h_int = x2_int - x1_int, y2_int - y1_int
            f_output_label.write(
                f'{frame_num},{id_},{x1_int:.2f},{y1_int:.2f},{w_int:.2f},{h_int:.2f},-1,-1,-1,-1\n'
            )
    f_output_label.close()
    sort.reset()
    return max_id, tracker_runtime_s


def main(args):

    fp_data = args.path_data
    
    list_seqs_names = sorted([i for i in os.listdir(fp_data) \
        if os.path.isdir(os.path.join(fp_data, i)) and i != '__pycache__'])
    assert len(list_seqs_names) > 0, f'ERROR: no sequence found in {fp_data}.'
    ip_runtime_log = os.path.join(fp_data, f'rtlog_{args.name_algo}.txt')
    if os.path.exists(ip_runtime_log):
        os.remove(ip_runtime_log)
    f_log = open(ip_runtime_log, 'w')
    f_log.write(f'Date of exp: {datetime.datetime.now()}; Algorithm: {args.name_algo}\n')
    f_log.write(
        'Seq_name,Num_frames,Tot_track,TrackerTime(s),TrackerRuntime(ms/frame),'
        'TrackerFPS,RunnerTime(s),RunnerRuntime(ms/frame),RunnerFPS\n'
    )

    for seq_name in list_seqs_names:
        ip_input_label = os.path.join(fp_data, seq_name, args.input_label_name)
        fp_frame = os.path.join(fp_data, seq_name, 'frame')
        assert os.path.exists(ip_input_label), f'ERROR: Input label file {ip_input_label} does not exist.'


        ip_output_label = os.path.join(fp_data, seq_name, f'result_{args.name_algo}.txt')
        if os.path.exists(ip_output_label):
            os.remove(ip_output_label)
        list_frames = sorted(glob.glob(os.path.join(fp_frame, '*.png')))
        assert len(list_frames) > 0, f'ERROR: no frames found in {fp_frame}.'
        assert os.path.basename(list_frames[0]) == 'img0001.png', \
            f'ERROR: the first frame should be img0001.png, but got {os.path.basename(list_frames[0])}.'
        num_frame = len(list_frames)
        last_labeled_frame = util.get_num_frame_from_label(ip_input_label)
        assert last_labeled_frame <= num_frame, \
            f'ERROR: label frame {last_labeled_frame} exceeds the {num_frame} image frames in {fp_frame}.'
        print(
            f'[INFO] Processing sequence {seq_name} with {num_frame} image frames '
            f'and labels through frame {last_labeled_frame}.'
        )
        st_frame_num, ed_frame_num = 1, num_frame
        assert ed_frame_num >= st_frame_num, f'ERROR: ed_frame_num {ed_frame_num} should be >= st_frame_num {st_frame_num}.'
        frame_sample = cv2.imread(list_frames[0])
        frame_size = (frame_sample.shape[1], frame_sample.shape[0])

        runner_time_start = time.perf_counter()
        max_id, tracker_runtime_s = sort_ori_process(
            ip_input_label, ip_output_label,
            st_frame_num, ed_frame_num, frame_size,
            args.sort_max_age, args.sort_min_hits, args.threshold, args.cost,
        )
        runner_runtime_s = time.perf_counter() - runner_time_start
        tracker_runtime_ms = tracker_runtime_s * 1000.0 / num_frame
        tracker_fps = num_frame / tracker_runtime_s if tracker_runtime_s > 0 else 0.0
        runner_runtime_ms = runner_runtime_s * 1000.0 / num_frame
        runner_fps = num_frame / runner_runtime_s if runner_runtime_s > 0 else 0.0
        f_log.write(
            f'{seq_name},{num_frame},{max_id},{tracker_runtime_s:.9f},'
            f'{tracker_runtime_ms:.6f},{tracker_fps:.6f},{runner_runtime_s:.9f},'
            f'{runner_runtime_ms:.6f},{runner_fps:.6f}\n'
        )
    f_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', type=str, default='dataset/omni_small/')
    parser.add_argument('--input_label_name', type=str, default='det.txt')
    parser.add_argument('--sort_max_age', type=int, default=10)
    parser.add_argument('--sort_min_hits', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--cost', type=str, default='iou',
                        choices=['iou', 'giou', 'omni_euc'],
                        help='Association cost for vanilla SORT (no SAMM).')
    parser.add_argument('--name_algo', type=str, default='sort_ori')

    args = parser.parse_args()

    main(args)


