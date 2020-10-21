import numpy as np
from utils.rectangle import Rectangle

from utils import api_utils


def non_maximum_suppression(boxes: np.ndarray, scores: np.ndarray, input_metadata, iou_thresh=1.0, score_thresh=0.001):
    iou_thresh = input_metadata.get('nmsThresh', 0.6)
    max_output_size = boxes.shape[0]

    D = np.empty((0, 4))
    while len(boxes) > 0:
        m = scores.argmax()
        M = Rectangle.from_coords(*boxes[m])
        boxes = np.delete(boxes, m, 0)
        D = np.vstack([D, M])
        for i, (box, score) in enumerate(zip(boxes, scores)):
            if M.iou(Rectangle.from_coords(*box)) >= iou_thresh:
                boxes = np.delete(boxes, i, 0)
                scores = np.delete(scores, i, 0)

    return D, scores


def multiclass_non_maximum_suppression(boxes, scores, classes, input_metadata):
    """
    perform multiclass non-maximum suppression on the given detections - discard detections that
    are already covered (high IoU) by another detection with higher score.
    the process is performed for every class separately, then the results are concatenated.

    :param boxes: [Nx4] np.array of boxes: x_min, y_min, x_max, y_max
    :param scores: [N] np.array of sorted scores (highest first)
    :param classes: [N] np.array of string class names
    :param input_metadata: a dictionary of additional input params. can be empty.
    :return: a filtered version of boxes, scores, classes
    """
    api_utils.check_correspondence(boxes, scores, classes)
    if len(scores) == 0:
        return boxes, scores, classes
    unique_classes = np.unique(classes)

    nms_boxes_list = np.empty((0, 4))
    nms_scores_list = np.empty((0,))
    nms_classes_list = np.empty((0,))

    for i, class_name in enumerate(unique_classes):
        class_indices = classes == class_name  # numpy syntax
        _boxes = boxes[class_indices]
        _scores = scores[class_indices]

        _boxes, _scores = non_maximum_suppression(_boxes, _scores, input_metadata)
        _classes = [class_name] * len(_scores)

        api_utils.check_correspondence(_boxes, _scores, _classes)

        nms_boxes_list = np.vstack((nms_boxes_list, _boxes))
        nms_scores_list = np.hstack((nms_scores_list, _scores))
        nms_classes_list = np.hstack((nms_classes_list, _classes))

    api_utils.check_correspondence(nms_boxes_list, nms_scores_list, nms_classes_list)
    return nms_boxes_list, nms_scores_list, nms_classes_list
