import os
import sys
import cv2
import datetime
import glob
import shutil
import numpy as np

sys.path.append('/home/xins/viterbi_motion_tracker/scripts')
import utils
from unit_sort_ori import Sort

import argparse


def sort_ori_process(
    ip_input_label, ip_output_label,
    st_frame_num, ed_frame_num, frame_size, 
    sort_max_age, sort_min_hits, threshold
):
    list_ip_frames = glob.glob(os.path.join(fp_frame, 'img*.png'))
    if os.path.exists(fp_result):
        shutil.rmtree(fp_result)
    os.makedirs(fp_result, exist_ok=True)
    os.makedirs(fp_result_label, exist_ok=True)

    sort = Sort(max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=threshold,)
    f_output_label = open(ip_output_label, 'w')

    max_id = -1
    dict_input_label = utils.load_input_label(ip_input_label, frame_size=frame_size)
    width, height = frame_size
    for frame_num in range(st_frame_num, ed_frame_num + 1):
        list_bbox = dict_input_label.get(frame_num, [])
        boxes = [bbox[1:] for bbox in list_bbox]
        gt_id_list = [bbox[0] for bbox in list_bbox]

        if len(boxes) == 0:
            boxes = np.empty((0, 5))
            gt_id_list = np.empty((0,))
        else:
            boxes = np.array(boxes)
        
        res = sort.update(boxes)
        boxes_track = res[:, :-1]
        boxes_ids = res[:, -1].astype(int)
        max_id = max(max_id, (max(boxes_ids) if len(boxes_ids) > 0 else -1))

        boxes_track_int = utils.box_frac_to_box_int(boxes_track, frame_size=frame_size)
        for box_track_int, box_track, id_ in zip(boxes_track_int, boxes_track, boxes_ids):
            x1, y1, x2, y2 = box_track
            x1_int, y1_int, x2_int, y2_int = box_track_int
            f_output_label.write(
                f'{frame_num},{id_},{x1_int:.2f},{y1_int:.2f},{x2_int:.2f},{y2_int:.2f},-1,-1,-1,-1\n')
    f_output_label.close()
    sort.reset()
    return max_id


def main(args):

    fp_data = args.path_data
    name_input_label = args.input_label_name
    
    list_seqs_names = sorted([i for i in os.listdir(fp_data) \
        if os.path.isdir(os.path.join(fp_data, i)) and i != '__pycache__'])
    ip_runtime_log = os.path.join(fp_data, f'rtlog_{name_algo}.txt')
    if os.path.exists(ip_runtime_log):
        os.remove(ip_runtime_log)
    f_log = open(ip_runtime_log, 'w')
    f_log.write(f'Date of exp: {datetime.datetime.now()}; Algorithm: {name_algo}\n')
    f_log.write(f'Seq_name,Num_frames,Tot_track,Runtime(ms),FPS\n')

    for seq_name in list_seqs_names:
        fp_frame = os.path.join(fp_data, seq_name, 'frame')
        ip_input_label = os.path.join(fp_data, seq_name, args.input_label_name)
        assert os.path.exists(ip_input_label), f'ERROR: Input label file {ip_input_label} does not exist.'

        ip_output_label = os.path.join(fp_data, seq_name, f'result_sort_ori.txt')
        if os.path.exists(ip_output_label):
            os.remove(ip_output_label)
        list_frames = sorted(glob.glob(os.path.join(fp_frame, '*.png')))
        assert len(list_frames) > 0, f'ERROR: no frames found in {fp_frame}.'
        assert os.path.basename(list_frames[0]) == 'img0001.png', \
            f'ERROR: the first frame should be img0001.png, but got {os.path.basename(list_frames[0])}.'
        st_frame_num, ed_frame_num = 1, len(list_frames)
        frame_sample = cv2.imread(list_frames[0])
        frame_size = (frame_sample.shape[1], frame_sample.shape[0])

        time_start = datetime.datetime.now()
        max_id = sort_ori_process(
            ip_input_label, ip_output_label,
            st_frame_num, ed_frame_num, frame_size,
            args.sort_max_age, args.sort_min_hits, args.threshold,
        )
        time_end = datetime.datetime.now()
        runtime = (time_end - time_start).microseconds / len(list_frames)
        fps = 1.0 / runtime if runtime > 0 else 0.0
        f_log.write(f'{seq_name},{len(list_frames)},{max_id},{runtime:.2f},{fps:.2f}\n')
    f_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_data', type=str, default='dataset/omni_small/')
    parser.add_argument('--input_label_name', type=str, default='det.txt')
    parser.add_argument('--sort_max_age', type=int, default=10)
    parser.add_argument('--sort_min_hits', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--name_algo', type=str, default='sort_ori')

    args = parser.parse_args()

    assert args.version == __version__, \
        f'Version mismatch: parser {args.version} vs code {__version__}'
    main(args)



