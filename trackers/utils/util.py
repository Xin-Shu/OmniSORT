import os
import sys
import glob
import numpy as np
import random
import colorsys

import lap
from scipy.optimize import linear_sum_assignment



def linear_assignment(cost_matrix):
    try:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0]) #

    except ImportError:
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def load_input_label(ip_input_label, frame_size, with_id=False, use_file_conf=False):
    """Load MOT-format labels as fractional [x1,y1,x2,y2,conf] boxes per frame.

    use_file_conf=False (default) hardcodes conf=1.0, matching how every
    OmniSORT baseline/grid runner has historically ingested detections. Set
    use_file_conf=True for score-aware trackers (ByteTrack/HybridSORT) on YOLOX
    detection inputs so the detector confidence (column 7) is preserved; a
    missing or negative score falls back to 1.0.
    """
    frame_width, frame_height = frame_size
    assert os.path.exists(ip_input_label), \
        f'ERROR: input label file {ip_input_label} does not exist.'
    dict_input_label = {}
    with open(ip_input_label, 'r') as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            num_frames = int(parts[0])
            oid = int(parts[1])
            x1 = float(parts[2])
            y1 = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            if use_file_conf and len(parts) > 6:
                raw_conf = float(parts[6])
                conf = raw_conf if raw_conf >= 0 else 1.0
            else:
                conf = 1.0
            x2 = x1 + w
            y2 = y1 + h
            if num_frames not in dict_input_label:
                dict_input_label[num_frames] = []
            if with_id:
                dict_input_label[num_frames].append(
                    [oid, x1 / frame_width, y1 / frame_height,
                          x2 / frame_width, y2 / frame_height, \
                     conf])
            else:
                dict_input_label[num_frames].append([x1 / frame_width, \
                    y1 / frame_height, x2 / frame_width, y2 / frame_height, \
                    conf])
    return dict_input_label

def box_frac_to_box_int(boxes_frac, frame_size):
    frame_width, frame_height = frame_size
    boxes_int = []
    for box_frac in boxes_frac:
        x1_frac, y1_frac, x2_frac, y2_frac, conf = box_frac
        x1_int = int(x1_frac * frame_width)
        y1_int = int(y1_frac * frame_height)
        x2_int = int(x2_frac * frame_width)
        y2_int = int(y2_frac * frame_height)
        boxes_int.append([x1_int, y1_int, x2_int, y2_int, conf])
    return boxes_int

def box_frac_to_box_int(list_box_frac, frame_size):
    img_width, img_height = frame_size
    list_box_int = []
    if len(list_box_frac) == 0:
        return np.array(list_box_int)
    if len(list_box_frac[0]) == 4:
        for box in list_box_frac:
            x1, y1, x2, y2 = box
            x1_int = int(x1 * img_width)
            y1_int = int(y1 * img_height)
            x2_int = int(x2 * img_width)
            y2_int = int(y2 * img_height)
            list_box_int.append([x1_int, y1_int, x2_int, y2_int])
    elif len(list_box_frac[0]) == 5:
        for box in list_box_frac:
            x1, y1, x2, y2, score = box
            x1_int = int(x1 * img_width)
            y1_int = int(y1 * img_height)
            x2_int = int(x2 * img_width)
            y2_int = int(y2 * img_height)
            list_box_int.append([x1_int, y1_int, x2_int, y2_int, score])
    else:
        raise ValueError("Each box in list_box_frac must have 4 or 5 elements.")
    return list_box_int

def get_num_frame_from_label(ip_input_label):
    assert os.path.exists(ip_input_label), \
        f'ERROR: input label file {ip_input_label} does not exist.'
    max_frame_num = -1
    with open(ip_input_label, 'r') as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            num_frames = int(parts[0])
            max_frame_num = max(max_frame_num, num_frames)
    return max_frame_num

