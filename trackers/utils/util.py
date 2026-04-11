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

def load_input_label(ip_input_label, frame_size):
    frame_width, frame_height = frame_size
    assert os.path.exists(ip_input_label), \
        f'ERROR: input label file {ip_input_label} does not exist.'
    dict_input_label = {}
    with open(ip_input_label, 'r') as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            num_frames = int(parts[0])
            id_ = int(parts[1])
            x1 = float(parts[2])
            y1 = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            if num_frames not in dict_input_label:
                dict_input_label[num_frames] = []
            dict_input_label[num_frames].append([id_, x1 / frame_width, \
                y1 / frame_height, w / frame_width, h / frame_height])
    return dict_input_label

def box_frac_to_box_int(boxes_frac, frame_size):
    frame_width, frame_height = frame_size
    boxes_int = []
    for box_frac in boxes_frac:
        x1_frac, y1_frac, x2_frac, y2_frac = box_frac
        x1_int = int(x1_frac * frame_width)
        y1_int = int(y1_frac * frame_height)
        x2_int = int(x2_frac * frame_width)
        y2_int = int(y2_frac * frame_height)
        boxes_int.append([x1_int, y1_int, x2_int, y2_int])
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
