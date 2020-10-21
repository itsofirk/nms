import argparse
import base64
import os
import os.path as osp

import cv2
from omek_tile_detector.service import model_manager
import numpy as np
import scipy.misc as misc
from omek_tile_detector.service.detection_api_detector import DetectionApiDetector
from omek_tile_detector.service.utils.detection_utils import run_clahe
from omek_tile_detector.service.omek_api import OmekApi
from omek_tile_detector.service.omek_handler import OmekHandler

from logic.detection_api_nms_performer import DetectionApiNmsPerformer
from utils import api_utils
from .nms_request_handler import NmsRequestHandler


def polylines(im, polys, color):
    return cv2.polylines(im, polys, True, color)


IMAGE_PATHS = []
ARGS = []


class Communicate:
    def __init__(self):
        print('initializing communicate object')
        class_name = ARGS[0].class_name
        self.class_name = class_name if class_name else os.environ['OMEK_CLASS']
        self.omek_handler = None
        self.init_omek_handler()

    def init_omek_handler(self):
        base64_model = model_manager.get_latest_model(self.class_name)
        label_map_string = model_manager.get_label_map_string(self.class_name)
        target_resolution = model_manager.get_target_resolution(self.class_name)
        target_pixel_size = model_manager.get_target_pixel_size(self.class_name)

        omek_detector = DetectionApiDetector(base64_model, label_map_string, target_resolution, target_pixel_size)
        omek_api = OmekApi()
        self.omek_handler = OmekHandler(omek_detector, omek_api)

    def loop(self, handle):
        for image_path in IMAGE_PATHS:
            print()
            print(image_path)

            ################
            #  omek        #
            ################
            args = ARGS[0]
            im = misc.imread(image_path)
            with open(image_path, 'rb') as f:
                im_encoded = f.read()
                im_base64 = base64.b64encode(im_encoded)
            original_resolution = 896 * 0.071 / im.shape[0]
            omek_request = {'image': im_base64, 'score_thresh': 0.2, 'top_k': 50, 'perform_clahe': False,
                            'geo_resolution': original_resolution}
            print('running detection')
            omek_results = self.omek_handler.detect(omek_request)

            if omek_results['statusType'] != 'progressReport':
                print(f'error in omek handler: \n {omek_results}', end="\n" * 3)
                continue

            ################
            #  nms         #
            ################
            print('performing nms')
            boxes, scores, classes = api_utils.unpack_detections(omek_results['detections'])
            # if len(np.unique(classes)) == 1:
            if self.omek_handler.omek_detector.num_classes == 1:
                classes[boxes[:, 2] > im.shape[1] / 2] = 'a'

            nms_request = {'detections': api_utils.pack_detections(boxes, scores, classes), 'iou_thresh': 0.1}
            nms_results = handle(nms_request)

            if omek_results['statusType'] != 'progressReport':
                print('error in nms handler:')
                print(nms_results)
                print('\n\n\n')
                continue

            nms_boxes, nms_scores, nms_classes = api_utils.unpack_detections(nms_results['detections'])

            ################
            #  vis         #
            ################
            if args.vis_dir is not None:
                if not osp.exists(args.vis_dir):
                    os.makedirs(args.vis_dir)
                vis_path = osp.join(args.vis_dir, osp.basename(image_path))

                colors = [(255, 255, 0), (0, 255, 255)]
                nms_colors = [(255, 127, 0), (0, 0, 225)]

                unique_classes = np.unique(classes)
                im_marked = im.copy()
                im_marked = run_clahe(im_marked)
                for i_cls, class_name in enumerate(unique_classes):
                    color = colors[i_cls]
                    curr_boxes = boxes[classes == class_name, :]
                    coords = map(api_utils.bbox2coords, curr_boxes[:100])
                    polys = map(lambda p: p.astype(np.int32).reshape((-1, 1, 2)), coords)
                    im_marked = polylines(im_marked, polys, color)

                    color = nms_colors[i_cls]
                    curr_boxes = nms_boxes[nms_classes == class_name, :]
                    coords = map(api_utils.bbox2coords, curr_boxes[:100])
                    polys = map(lambda p: p.astype(np.int32).reshape((-1, 1, 2)), coords)
                    im_marked = polylines(im_marked, polys, color)
                misc.imsave(vis_path, im_marked)

        print()


def parse_args():
    """
    arse input arguments
    :return: an argparse argument Namespace.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--vis_dir', type=str, default=None,
                        help='optional. visualizations dir')
    parser.add_argument('--class_name', type=str, default=None,
                        help='class names. default behavior is reading from environment variable OMEK_CLASS')
    parser.add_argument('out_dir', type=str,
                        help='dir to write result CSVs into')
    parser.add_argument('image_paths', nargs=argparse.REMAINDER,
                        help='images to run detection and NMS on')

    args = parser.parse_args()
    return args


def mock():
    args = parse_args()
    IMAGE_PATHS.extend(args.image_paths)
    ARGS.append(args)

    nms_performer = DetectionApiNmsPerformer()
    nms_performer.create_session()
    nms_handler = NmsRequestHandler(nms_performer)
    handle = nms_handler  # .nms

    comm = Communicate()
    comm.loop(handle)
