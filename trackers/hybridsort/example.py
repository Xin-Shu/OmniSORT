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
from hybrid_sort import Hybrid_Sort
import argparse


class HybridSortArgs:
    """Minimal args container expected by Hybrid_Sort / KalmanBoxTracker.

    Only the attributes actually read on the no-ReID path are exposed:
    the score-modulation (TCM) toggles + weights and track_thresh, which
    the score Kalman filter clips its confidence against.
    """

    def __init__(self, track_thresh=0.6,
                 TCM_first_step=True, TCM_byte_step=True,
                 TCM_first_step_weight=1.0, TCM_byte_step_weight=1.0):
        self.track_thresh = track_thresh
        self.TCM_first_step = TCM_first_step
        self.TCM_byte_step = TCM_byte_step
        self.TCM_first_step_weight = TCM_first_step_weight
        self.TCM_byte_step_weight = TCM_byte_step_weight


def hybridsort_process(
    hs_args, ip_input_label, ip_output_label,
    st_frame_num, ed_frame_num, frame_size,
    max_age, min_hits, iou_threshold, det_thresh,
    delta_t, asso_func, inertia, use_byte,
    use_det_conf=False,
):
    tracker = Hybrid_Sort(
        args=hs_args,
        det_thresh=det_thresh,
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        delta_t=delta_t,
        asso_func=asso_func,
        inertia=inertia,
        use_byte=use_byte,
    )
    f_output_label = open(ip_output_label, 'w')

    max_id = -1
    tracker_runtime_s = 0.0
    dict_input_label = util.load_input_label(
        ip_input_label, frame_size=frame_size, use_file_conf=use_det_conf)
    for frame_num in range(st_frame_num, ed_frame_num + 1):
        list_bbox = dict_input_label.get(frame_num, [])
        list_bbox_int = util.box_frac_to_box_int(list_bbox, frame_size)
        dets = np.array(list_bbox_int, dtype=float)

        if dets.size == 0:
            dets = np.empty((0, 5), dtype=float)

        img_info = (frame_size[1], frame_size[0], 1.0)
        img_size = (frame_size[1], frame_size[0])
        tracker_time_start = time.perf_counter()
        tracks = tracker.update(dets, img_info, img_size)
        tracker_runtime_s += time.perf_counter() - tracker_time_start

        for t in tracks:
            t = np.asarray(t).tolist()
            if len(t) < 5:
                continue
            x1, y1, x2, y2, track_id = t[:5]
            track_id = int(track_id)
            max_id = max(max_id, track_id)
            w = x2 - x1
            h = y2 - y1
            f_output_label.write(
                f'{frame_num},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},-1,-1,-1,-1\n'
            )
    f_output_label.close()
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

    hs_args = HybridSortArgs(
        track_thresh=args.track_thresh,
        TCM_first_step=args.TCM_first_step,
        TCM_byte_step=args.TCM_byte_step,
        TCM_first_step_weight=args.TCM_first_step_weight,
        TCM_byte_step_weight=args.TCM_byte_step_weight,
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
        max_id, tracker_runtime_s = hybridsort_process(
            hs_args, ip_input_label, ip_output_label,
            st_frame_num, ed_frame_num, frame_size,
            args.max_age, args.min_hits, args.threshold, args.det_thresh,
            args.delta_t, args.asso_func, args.inertia, args.use_byte,
            use_det_conf=args.use_det_conf,
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
    parser.add_argument('--max_age', type=int, default=10)
    parser.add_argument('--min_hits', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='IoU/association threshold for first matching')
    parser.add_argument('--det_thresh', type=float, default=0.3)
    parser.add_argument('--delta_t', type=int, default=3)
    parser.add_argument('--asso_func', type=str, default='hmiou',
                        choices=['iou', 'giou', 'ciou', 'diou', 'ct_dist', 'hmiou'])
    parser.add_argument('--inertia', type=float, default=0.2)
    parser.add_argument('--use_byte', action='store_true')
    parser.add_argument('--use_det_conf', action='store_true',
                        help='Use detector confidence (col 7) as detection score; '
                             'for YOLOX detection inputs. GT inputs stay conf=1.0.')
    parser.add_argument('--track_thresh', type=float, default=0.6)
    parser.add_argument('--TCM_first_step', action='store_true', default=True)
    parser.add_argument('--TCM_byte_step', action='store_true', default=True)
    parser.add_argument('--TCM_first_step_weight', type=float, default=1.0)
    parser.add_argument('--TCM_byte_step_weight', type=float, default=1.0)
    parser.add_argument('--name_algo', type=str, default='hybridsort_ori')

    args = parser.parse_args()

    main(args)
