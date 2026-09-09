"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import numpy as np
from association import *


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age-dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h+1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])# if x[2] * x[3] > 0 else 0
    h = x[2] / w if w != 0 else x[2]
    if(score == None):
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
    cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm


def signed_principal_wrap(value, period=1.0):
    """Map a periodic value to the signed principal interval."""
    return ((value + period / 2.0) % period) - period / 2.0


def horizontal_velocity_change_ratio(previous_velocity, updated_velocity, epsilon=1e-6):
    return np.abs(updated_velocity - previous_velocity) / np.maximum(np.abs(previous_velocity), epsilon)


def horizontal_velocity_change_exceeds_threshold(previous_velocity, updated_velocity, threshold, epsilon=1e-6):
    return bool(np.any(horizontal_velocity_change_ratio(previous_velocity, updated_velocity, epsilon) > threshold))


def correct_horizontal_velocity(previous_velocity, updated_velocity, threshold, epsilon=1e-6):
    if horizontal_velocity_change_exceeds_threshold(previous_velocity, updated_velocity, threshold, epsilon):
        return signed_principal_wrap(updated_velocity)
    return updated_velocity


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, thres_de_velo=10.0, delta_t=3, speed_correction_method='signed_principal_wrap'):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        from filterpy.kalman import KalmanFilter
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0], 
            [0, 1, 0, 0, 0, 1, 0, 0], 
            [0, 0, 1, 0, 0, 0, 1, 0], 
            [0, 0, 0, 1, 0, 0, 0, 1],  
            [0, 0, 0, 0, 1, 0, 0, 0], 
            [0, 0, 0, 0, 0, 1, 0, 0], 
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0], 
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0], 
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        # self.history = []
        self.history = [np.array(bbox[:4]).reshape((1, 4))]
        self.hits = 0
        self.hit_streak = 0
        self.thres_de_velo = thres_de_velo
        self.speed_correction_method = speed_correction_method
        assert self.speed_correction_method in ['prev_speed', 'signed_principal_wrap', 're_update'], \
            f"Speed correction method {self.speed_correction_method} not recognized."
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        if bbox is not None:
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    dt = self.delta_t - i
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age-dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \\Delta t steps away
                """
                self.velocity = speed_direction(previous_box, bbox)
            
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1

            _prev_x_speed_, _prev_y_speed_ = self.kf.x[4].copy(), self.kf.x[5].copy()
            self.kf.update(convert_bbox_to_z(bbox))
            _upda_x_speed_, _upda_y_speed_ = self.kf.x[4], self.kf.x[5]
            _if_correct_x_speed_ = horizontal_velocity_change_exceeds_threshold(
                _prev_x_speed_, _upda_x_speed_, self.thres_de_velo
            )
            
            if self.speed_correction_method == 'prev_speed':
                self.kf.x[4] = _prev_x_speed_ if _if_correct_x_speed_ else _upda_x_speed_
            elif self.speed_correction_method == 'signed_principal_wrap':
                self.kf.x[4] = correct_horizontal_velocity(
                    _prev_x_speed_, _upda_x_speed_, self.thres_de_velo
                )
            elif self.speed_correction_method == 're_update':
                if _if_correct_x_speed_ and _prev_x_speed_ > 0:
                    bbox[0] = bbox[0] + 1.0
                    bbox[2] = bbox[2] + 1.0
                elif _if_correct_x_speed_ and _prev_x_speed_ < 0:
                    bbox[0] = bbox[0] - 1.0
                    bbox[2] = bbox[2] - 1.0
                self.kf.update(convert_bbox_to_z(bbox))
            
            self.kf.x[:4] = convert_bbox_to_z(bbox)
            self.history.append(convert_x_to_bbox(self.kf.x))
        else:
            self.kf.update(bbox)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()

        self.kf.x[0] = self.kf.x[0] % 1.0

        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        # return convert_x_to_bbox(self.kf.x)
        return self.history[-1]


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {  "iou": iou_batch,
                "giou": giou_batch,
                "ciou": ciou_batch,
                "diou": diou_batch,
                "euc": ct_dist,
                "omni_euc": omnieuc_batch,}


class OmniOCSORT(object):
    def __init__(self, det_thresh, thres_de_velo=10.0, max_age=30, min_hits=3, speed_correction_method='signed_principal_wrap',
        iou_threshold=0.3, delta_t=3, asso_func="iou", list_weights=[], inertia=0.2, use_byte=False):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.thres_de_velo = thres_de_velo
        self.speed_correction_method = speed_correction_method
        self.delta_t = delta_t
        self.asso_func = [ASSO_FUNCS[af] for af in asso_func] if isinstance(asso_func, list) else [ASSO_FUNCS[asso_func]]
        self.inertia = inertia
        self.use_byte = use_byte
        self.list_weights = list_weights
        if len(self.list_weights) != len(self.asso_func):
            self.list_weights = [1 / len(self.asso_func)] * len(self.asso_func)
        KalmanBoxTracker.count = 0
    
    def weighted_association(self, dets, trks):

        tot_score = None
        for i in range(len(self.asso_func)):
            asso_func = self.asso_func[i]
            weight = self.list_weights[i]
            cost_met = asso_func(dets, trks)
            if tot_score is None:
                tot_score = weight * cost_met
            else:
                tot_score += weight * cost_met
        return tot_score

    def update(self, output_results, img_info, img_size):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if output_results is None:
            return np.empty((0, 5))

        self.frame_count += 1
        # post_process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]

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

        velocities = np.array(
            [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        e_fuse = self.weighted_association(dets, trks)
        matched, unmatched_dets, unmatched_trks = associate_efuse(
            dets, trks, e_fuse, self.iou_threshold, velocities, k_observations, self.inertia)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        """
            Second round of associaton by OCR
        """
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            e_fuse = self.weighted_association(dets_second, u_trks)
            e_fuse = np.array(e_fuse)
            if e_fuse.min() < self.iou_threshold:
                matched_indices = linear_assignment(e_fuse)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if e_fuse[m[0], m[1]] > self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets_second[det_ind, :])
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            e_fuse = self.weighted_association(left_dets, left_trks)
            e_fuse = np.array(e_fuse)
            if e_fuse.min() < self.iou_threshold:
                rematched_indices = linear_assignment(e_fuse)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if e_fuse[m[0], m[1]] > self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], speed_correction_method=self.speed_correction_method, \
                delta_t=self.delta_t, thres_de_velo=self.thres_de_velo)
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))
