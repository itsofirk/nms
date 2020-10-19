import numpy as np
import tensorflow as tf
from object_detection.core.post_processing import multiclass_non_max_suppression
from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.python.client.session import Session

SCORE_THRESH = 0.001


class DetectionApiNmsPerformer:
    def __init__(self):
        self.sess = None
        self.config = ConfigProto()
        self.config.gpu_options.allow_growth = True

    def create_session(self):
        """
        create tensorflow session
        """
        self.sess = Session(config=self.config)

    def nms_single_class(self, boxes, scores, input_metadata):
        """
        Single-class NMS implementation using Tensorflow framework
        perform non-maximum suppression on the given detections - discard detections that
        are already covered (high IoU) by another detection with higher score.
        :param boxes:
        :param scores:
        :param input_metadata:
        :return: a filtered version of boxes, scores
        """
        # define nms function params
        # This is not supposed to be in the request! we insert a default value for safety.
        score_thresh = SCORE_THRESH
        iou_thresh = input_metadata.get('nmsThresh', 0.6)
        max_output_size = boxes.shape[0]

        # expand to adapt to expected input dimensions
        boxes = boxes[:, np.newaxis, :]
        scores = scores[:, np.newaxis]

        # convert to tensors
        boxes = tf.constant(boxes, tf.float32)
        scores = tf.constant(scores, tf.float32)

        # create nms graph node
        nms_node = multiclass_non_max_suppression(
            boxes, scores, score_thresh, iou_thresh, max_output_size)

        boxlist, tensor = nms_node
        # run nms evaluation
        self.sess.run(tensor)
        nms_boxes = boxlist.data['boxes'].eval(session=self.sess)
        nms_scores = boxlist.data['scores'].eval(session=self.sess)

        return nms_boxes, nms_scores

    def suppress(self, boxes, scores, classes, input_metadata):
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
