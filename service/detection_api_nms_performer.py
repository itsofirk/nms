import numpy as np
import tensorflow as tf
from object_detection.core.post_processing import multiclass_non_max_suppression
from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.python.client.session import Session

from .abstracts import NmsPerformerBase

# from .object_detection.core.post_processing import multiclass_non_max_suppression

SCORE_THRESH = 0.001


class DetectionApiNmsPerformer(NmsPerformerBase):
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
        """ see NmsPerformerBase """
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

        # run nms evaluation
        self.sess.run(nms_node.get())
        nms_boxes = nms_node.data['boxes'].eval(session=self.sess)
        nms_scores = nms_node.data['scores'].eval(session=self.sess)

        return nms_boxes, nms_scores
