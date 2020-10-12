import abc
import numpy as np


class NmsPerformerBase:
    @abc.abstractmethod
    def nms_single_class(self, boxes, scores, input_metadata):
        """
        perform non-maximum suppression on the given detections - discard detections that
        are already covered (high IoU) by another detection with higher score.
        :param boxes:
        :param scores:
        :param input_metadata:
        :return: a filtered version of boxes, scores
        """
        return boxes, scores

    def suppress(self, boxes, scores, classes, input_metadata):
        """
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
        nms_boxes_list = [None] * len(unique_classes)
        nms_scores_list = [None] * len(unique_classes)
        nms_classes_list = [None] * len(unique_classes)

        for i_class, class_name in enumerate(unique_classes):
            relevant_inds = classes == class_name
            curr_boxes = boxes[relevant_inds]
            curr_scores = scores[relevant_inds]

            curr_nms_boxes, curr_nms_scores = self.nms_single_class(curr_boxes, curr_scores, input_metadata)
            curr_nms_classes = np.array([class_name] * len(curr_nms_scores))

            nms_boxes_list[i_class] = curr_nms_boxes
            nms_scores_list[i_class] = curr_nms_scores
            nms_classes_list[i_class] = curr_nms_classes

        if len(nms_boxes_list) > 0:
            nms_boxes = np.vstack(nms_boxes_list)
            nms_scores = np.hstack(nms_scores_list)
            nms_classes = np.hstack(nms_classes_list)
        else:
            nms_boxes = np.array(nms_boxes_list)
            nms_scores = np.array(nms_scores_list)
            nms_classes = np.array(nms_classes_list)

        return nms_boxes, nms_scores, nms_classes
