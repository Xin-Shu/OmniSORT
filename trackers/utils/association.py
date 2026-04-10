import numpy as np


def iou_batch_ori(bb_test, bb_gt):
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
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
    return (o) 

def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]

    *Edit for OmniSORT*: reverse the IoU score so that lower IoU 
        corresponds to better match and higher IoU corresponds to worse match.
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

def omnieuc_batch(bb_observe, bb_predict):
    """
    Seam-Aware Euclidean Distance, that measures the distance between 
        two bounding boxes from an omnidirectional footage. Coordinates
        are normalized to [0, 1] and are used to compute the minimum 
        between the direct centre distance and its wrapped counterpart 
        obtained by shifting by one image width (and height when 
        vertical wrapping is enabled)

    Args:
        bb_predict: [x1, y1, x2, y2] in normalized 
            coordinates (0 to 1), predicted by the tracker
        bb_observe: [x1, y1, x2, y2] in normalized 
            coordinates (0 to 1), observed from the detection results
        return: OmniEuc distance between bb_predict 
            and bb_observe, normalized to [0, 1]
    """
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
    omnieuc = np.minimum(euclidean, omni_euclidean)

    return (omnieuc - omnieuc.min()) / max((omnieuc.max() - omnieuc.min()), 1e-6)

def giou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return: GIoU distance between bboxes1 and bboxes2, normalized to [0, 1]

    *Edit for OmniSORT*: reverse the GIoU score so that lower GIoU 
        corresponds to better match and higher GIoU corresponds to worse match.
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
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



