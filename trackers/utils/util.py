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