import os
import sys
import cv2
import datetime
import glob
import shutil
import numpy as np

sys.path.append('../utils/')
import util
from ocsort import OCSort
import argparse


def ocsort_ori_process(
    ip_input_label, ip_output_label, asso_func,
    st_frame_num, ed_frame_num, frame_size, 
    max_age, min_hits, threshold, det_thresh,
):
    ocsort = OCSort(
        max_age=max_age, 
        min_hits=min_hits, 
        iou_threshold=threshold,
        asso_func=asso_func,
        det_thresh=det_thresh,
    )
    f_output_label = open(ip_output_label, 'w')

    max_id = -1
    dict_input_label = util.load_input_label(ip_input_label, frame_size=frame_size)
    for frame_num in range(st_frame_num, ed_frame_num + 1):
        list_bbox = dict_input_label.get(frame_num, [])
        list_bbox_int = util.box_frac_to_box_int(list_bbox, frame_size)
        dets = np.array(list_bbox_int)

        if dets.size == 0:
            dets = np.empty((0, 5), dtype=float)
        
        img_info = (frame_size[1], frame_size[0], 1.0)
        img_size = (frame_size[1], frame_size[0])
        tracks = ocsort.update(dets, img_info, img_size)

        for t in tracks:
            x1, y1, x2, y2, track_id = t
            max_id = max(max_id, int(track_id))
            w = x2 - x1
            h = y2 - y1
            f_output_label.write(
                f'{frame_num},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},-1,-1,-1,-1\n'
            )
    f_output_label.close()
    return max_id


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
    f_log.write(f'Seq_name,Num_frames,Tot_track,Runtime(ms),FPS\n')

    for seq_name in list_seqs_names:
        ip_input_label = os.path.join(fp_data, seq_name, args.input_label_name)
        num_frame = util.get_num_frame_from_label(ip_input_label)
        print(f'[INFO] Processing sequence {seq_name} with {num_frame} frames.')
        fp_frame = os.path.join(fp_data, seq_name, 'frame')
        assert os.path.exists(ip_input_label), f'ERROR: Input label file {ip_input_label} does not exist.'


        ip_output_label = os.path.join(fp_data, seq_name, f'result_{args.name_algo}.txt')
        if os.path.exists(ip_output_label):
            os.remove(ip_output_label)
        list_frames = sorted(glob.glob(os.path.join(fp_frame, '*.png')))
        assert len(list_frames) > 0, f'ERROR: no frames found in {fp_frame}.'
        assert os.path.basename(list_frames[0]) == 'img0001.png', \
            f'ERROR: the first frame should be img0001.png, but got {os.path.basename(list_frames[0])}.'
        st_frame_num, ed_frame_num = 1, num_frame
        assert ed_frame_num > st_frame_num, f'ERROR: ed_frame_num {ed_frame_num} should be >= st_frame_num {st_frame_num}.'
        frame_sample = cv2.imread(list_frames[0])
        frame_size = (frame_sample.shape[1], frame_sample.shape[0])

        time_start = datetime.datetime.now()
        max_id = ocsort_ori_process(
            ip_input_label, ip_output_label, args.asso_func,
            st_frame_num, ed_frame_num, frame_size,
            args.max_age, args.min_hits, args.threshold, args.det_thresh,
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
    parser.add_argument('--max_age', type=int, default=10)
    parser.add_argument('--min_hits', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--det_thresh', type=float, default=0.3)
    parser.add_argument('--name_algo', type=str, default='sort_ori')
    parser.add_argument('--asso_func', type=str, default='iou')

    args = parser.parse_args()

    main(args)



