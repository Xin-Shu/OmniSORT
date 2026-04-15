import numpy as np
import lap
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):

    try:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0]) #

    except ImportError:
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def giou_batch(bboxes1, bboxes2):
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    union = ((bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])                                      
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) - wh)  
    iou = wh / union

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    wc = xxc2 - xxc1 
    hc = yyc2 - yyc1 
    # assert((wc > 0).all() and (hc > 0).all())
    area_enclose = wc * hc 
    giou = iou - (area_enclose - union) / area_enclose
    giou = (giou + 1.) / 2.0 # resize from (-1,1) to (0,1)
    return 1 - giou

def omnieuc_batch(bb_observe, bb_predict):
    bb_predict = np.array(bb_predict) 
    bb_observe = np.array(bb_observe)

    bb_predict = np.expand_dims(bb_predict, 0)
    bb_observe = np.expand_dims(bb_observe, 1)

    bb_observe_center_x = (bb_observe[..., 0] + bb_observe[..., 2]) / 2
    bb_observe_center_y = (bb_observe[..., 1] + bb_observe[..., 3]) / 2
    bb_predict_center_x = (bb_predict[..., 0] + bb_predict[..., 2]) / 2
    bb_predict_center_y = (bb_predict[..., 1] + bb_predict[..., 3]) / 2

    euclidean = np.sqrt(
        (bb_observe_center_x - bb_predict_center_x) ** 2 +
        (bb_observe_center_y - bb_predict_center_y) ** 2
    ) / np.sqrt(2)

    dx = np.abs(bb_observe_center_x - bb_predict_center_x)
    dy = np.abs(bb_observe_center_y - bb_predict_center_y)

    dx_omni = np.minimum(dx, 1.0 - dx)
    dy_omni = np.minimum(dy, 1.0 - dy)

    omni_euclidean = np.sqrt(dx_omni ** 2 + dy_omni ** 2) / np.sqrt(2)
    omni_euc = np.minimum(euclidean, omni_euclidean)

    if omni_euc.shape[0] == 0:
        return omni_euc
    return (omni_euc - omni_euc.min()) / max((omni_euc.max() - omni_euc.min()), 1e-6)

def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) \
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)  
    o = 1 - o       
    return o  

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(abs(x[2] * x[3]))
    h = x[2] / w if w != 0 else x[2]
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., \
            x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., \
            x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(
        self, 
        bbox, thres_de_velo=2.5, speed_correction_method='prev_speed',
        if_use_kalman_loc=False, oid=None, if_omni=False
    ):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=8, dim_z=4) 
        self.thres_de_velo = thres_de_velo
        self.speed_correction_method = speed_correction_method
        assert self.speed_correction_method in ['prev_speed', 'mod_by_1', 'mod_by_1_plus_0.5', 're_update'], \
            f"Speed correction method {self.speed_correction_method} not recognized."
        self.kf.F = np.array([
            [1., 0., 0., 0., 1., 0., 0., 0.],
            [0., 1., 0., 0., 0., 1., 0., 0.],
            [0., 0., 1., 0., 0., 0., 1., 0.],
            [0., 0., 0., 1., 0., 0., 0., 1.],
            [0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1.]], dtype=np.float32)
        self.kf.H = np.array([
            [1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0.]], dtype=np.float32)

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        if oid is None:
            self.id = KalmanBoxTracker.count
        else:
            if oid in range(KalmanBoxTracker.count):
                raise warning(f"Object ID {oid} already exists. Please provide a unique ID.")
            else:
                self.id = oid
        KalmanBoxTracker.count += 1
        self.history = [np.array(bbox[:4]).reshape((1, 4))]
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.if_use_kalman_loc = if_use_kalman_loc
        self.if_omni = if_omni

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        _prev_x_speed_, _prev_y_speed_ = self.kf.x[4], self.kf.x[5]
        self.kf.update(convert_bbox_to_z(bbox))
        _upda_x_speed_, _upda_y_speed_ = self.kf.x[4], self.kf.x[5]
        _perc_x_acce_ = abs((_upda_x_speed_ - _prev_x_speed_) / _prev_x_speed_) if _prev_x_speed_ != 0 else 0
        _perc_y_acce_ = abs((_upda_y_speed_ - _prev_y_speed_) / _prev_y_speed_) if _prev_y_speed_ != 0 else 0

        if self.speed_correction_method == 'prev_speed':
            self.kf.x[4] = _prev_x_speed_ if _perc_x_acce_ > self.thres_de_velo else _upda_x_speed_
            self.kf.x[5] = _prev_y_speed_ if _perc_y_acce_ > self.thres_de_velo else _upda_y_speed_
        elif self.speed_correction_method == 'mod_by_1':
            self.kf.x[4] = ((_upda_x_speed_) % 1.0) if _perc_x_acce_ > self.thres_de_velo else _upda_x_speed_
            self.kf.x[5] = ((_upda_y_speed_) % 1.0) if _perc_y_acce_ > self.thres_de_velo else _upda_y_speed_
        elif self.speed_correction_method == 'mod_by_1_plus_0.5':
            self.kf.x[4] = ((_upda_x_speed_ + 0.5) % 1.0) - 0.5 if _perc_x_acce_ > self.thres_de_velo else _upda_x_speed_
            self.kf.x[5] = ((_upda_y_speed_ + 0.5) % 1.0) - 0.5 if _perc_y_acce_ > self.thres_de_velo else _upda_y_speed_
        elif self.speed_correction_method == 're_update':
            if (_perc_x_acce_ > self.thres_de_velo) and _prev_x_speed_ > 0:
                bbox[0] = bbox[0] + 1.0
                bbox[2] = bbox[2] + 1.0
            elif (_perc_x_acce_ > self.thres_de_velo) and _prev_x_speed_ < 0:
                bbox[0] = bbox[0] - 1.0
                bbox[2] = bbox[2] - 1.0
            if (_perc_y_acce_ > self.thres_de_velo) and _prev_y_speed_ > 0:
                bbox[1] = bbox[1] + 1.0
                bbox[3] = bbox[3] + 1.0
            elif (_perc_y_acce_ > self.thres_de_velo) and _prev_y_speed_ < 0:
                bbox[1] = bbox[1] - 1.0
                bbox[3] = bbox[3] - 1.0
            self.kf.update(convert_bbox_to_z(bbox))

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        if self.if_use_kalman_loc:
            self.history.append(convert_x_to_bbox(self.kf.x))
        else:
            self.history.append(np.array(bbox[:4]).reshape((1, 4)))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()

        self.kf.x[0] = self.kf.x[0] % 1.0 if self.if_omni else self.kf.x[0]
        self.kf.x[1] = self.kf.x[1] % 1.0 if self.if_omni else self.kf.x[1]

        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
            Returns the current bounding box estimate in the format of 
                [x1,y1,x2,y2] with a shape of (1, 4)
        """
        return self.history[-1]


def associate_detections_to_trackers(
        detections, trackers, list_cost_types, list_weights=[], 
        threshold=0.3, if_omni=False, frame_num=None):
    """
        Assigns detections to tracked object (both represented as bounding boxes)
        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    cost_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    if len(list_weights) == 0 or len(list_weights) != len(list_cost_types):
        if 'iou' in list_cost_types:
            cost_matrix += iou_batch(detections, trackers)
        if 'euc' in list_cost_types:
            cost_matrix += omnieuc_batch(detections, trackers,)
        if 'giou' in list_cost_types:
            cost_matrix += giou_batch(detections, trackers)
        cost_matrix = cost_matrix / len(list_cost_types)
    elif len(list_weights) == len(list_cost_types):
        weight_sum = sum(list_weights)
        for idx, cost_type in enumerate(list_cost_types):
            weight = list_weights[idx] / weight_sum
            if cost_type == 'iou':
                cost_matrix += weight * iou_batch(detections, trackers)
            if cost_type == 'euc':
                cost_matrix += weight * omnieuc_batch(detections, trackers,)
            if cost_type == 'giou':
                cost_matrix += weight * giou_batch(detections, trackers)

    if min(cost_matrix.shape) > 0:
        a = (cost_matrix < threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(cost_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if(cost_matrix[m[0], m[1]] > threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches) == 0):
        matches = np.empty((0,2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class OmniSort(object):
    def __init__(self, 
            max_age, min_hits, threshold, thres_de_velo=2.5,
            speed_correction_method='prev_speed',
            list_cost_types=[], list_weights=[],
            if_omni=True, 
            if_use_kalman_loc=False
        ):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.threshold = threshold
        self.trackers = []
        self.frame_count = 0
        self.if_omni = if_omni
        self.if_use_kalman_loc = if_use_kalman_loc
        self.list_cost_types = list_cost_types
        self.list_weights = list_weights
        self.thres_de_velo = thres_de_velo
        self.speed_correction_method = speed_correction_method

        assert len(self.list_cost_types) > 0, "At least one cost type must be specified in list_cost_types."

    def update(self, dets=np.empty((0, 5)), frame=None, oid=None):
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        # print('Frame: ', self.frame_count)
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = \
            associate_detections_to_trackers(
                dets, trks, self.list_cost_types, self.list_weights,
                threshold=self.threshold, 
                if_omni=self.if_omni, frame_num=self.frame_count
            )

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[i, :], thres_de_velo=self.thres_de_velo, 
                speed_correction_method=self.speed_correction_method,
                if_use_kalman_loc=self.if_use_kalman_loc, oid=oid, if_omni=self.if_omni)
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1)) # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)

        if(len(ret) > 0):
            return np.concatenate(ret)

        return np.empty((0, 5))

    def reset(self):
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0