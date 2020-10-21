import numpy as np
from utils.rectangle import Rectangle


def non_maximum_suppression(boxes: np.ndarray, scores: np.ndarray, input_metadata, iou_thresh=1.0, score_thresh=0.001):
    iou_thresh = input_metadata.get('nmsThresh', 0.6)
    max_output_size = boxes.shape[0]

    D = np.empty((0, 4))  # init results list
    while len(boxes) > 0:
        # find the highest score index
        m = scores.argmax()
        # assign the corresponding box
        M = Rectangle.from_coords(*boxes[m])
        # add to final list and remove from candidates
        boxes = np.delete(boxes, m, 0)
        D = np.vstack([D, M])
        # find similar boxes with overlap > threshold
        for i, (box, score) in enumerate(zip(boxes, scores)):
            if M.iou(Rectangle.from_coords(*box)) >= iou_thresh:
                # if so, disqualify
                boxes = np.delete(boxes, i, 0)
                scores = np.delete(scores, i, 0)

    return D, scores


def multiclass_non_maximum_supression(self, boxes, scores, classes, input_metadata):
    """
    Multi-class NMS implementation
    perform non-maximum suppression on the given detections - discard detections that
    are already covered (high IoU) by another detection with higher score.
    the process is performed for every class separately, then the results are concatenated.

    :param boxes: [Nx4] np.array of boxes: x_min, y_min, x_max, y_max
    :param scores: [N] np.array of sorted scores (highest first)
    :param classes: [N] np.array of string class names
    :param input_metadata: a dictionary of additional input params. can be empty.
    :return: a filtered version of boxes, scores, classes
    """
    unique_classes = np.unique(classes)
    nms_boxes_list = np.full(len(unique_classes), None)
    nms_scores_list = np.full(len(unique_classes), None)
    nms_classes_list = np.full(len(unique_classes), None)

    for idx, class_name in enumerate(unique_classes):
        relevant_inds = classes == class_name  # numpy syntax
        curr_boxes = boxes[relevant_inds]
        curr_scores = scores[relevant_inds]

        curr_nms_boxes, curr_nms_scores = self.nms_single_class(curr_boxes, curr_scores, input_metadata)
        curr_nms_classes = np.array([class_name] * len(curr_nms_scores))

        nms_boxes_list[idx] = curr_nms_boxes
        nms_scores_list[idx] = curr_nms_scores
        nms_classes_list[idx] = curr_nms_classes

    if len(nms_boxes_list) > 0:
        nms_boxes = np.vstack(nms_boxes_list)
        nms_scores = np.hstack(nms_scores_list)
        nms_classes = np.hstack(nms_classes_list)
    else:
        nms_boxes = np.array(nms_boxes_list)
        nms_scores = np.array(nms_scores_list)
        nms_classes = np.array(nms_classes_list)

    return nms_boxes, nms_scores, nms_classes
