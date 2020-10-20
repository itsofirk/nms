import numpy as np
from utils.rectangle import Rectangle


def non_maximum_supression(boxes: np.ndarray, scores: np.ndarray, iou_thresh, input_metadata, score_thresh=0.001):
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


def multiclass_non_maximum_supression():
    pass
