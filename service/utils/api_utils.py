import numpy as np
import pandas as pd
from StringIO import StringIO
import traceback

from nms.service.object_detection.utils.np_box_ops import bbox2coords, coords2bbox


def prepare_detections_json(boxes, scores, classes):
    """
    structure detections into json-style format as defined in the swagger file
    """
    detections = []
    for box, score, cls in zip(boxes, scores, classes):
        coords = bbox2coords(box)
        outer_ring = coords.tolist()
        pixel_polygon = {
            'type': 'Polygon',
            'coordinates': [outer_ring]
        }
        detection = {
            'classification': cls,
            'grade': float(score),
            'pixelLocation': pixel_polygon
        }
        detections.append(detection)
    return detections


def parse_detections_json(detections):
    """
    parse detections from the json-style format define in the swagger file into boxes, scores, classes
    """
    coords = map(lambda x: np.array(x['pixelLocation']['coordinates'][0]), detections)
    boxes = np.array(map(coords2bbox, coords))
    scores = np.array(map(lambda x: x['grade'], detections)).astype('float32')
    classes = np.array(map(lambda x: x['classification'], detections))
    return boxes, scores, classes


def prepare_detections_csv(boxes, scores, classes):
    """
    structure detections into csv format
    """
    df = pd.DataFrame({
        'X1': boxes[:, 0],
        'Y1': boxes[:, 1],
        'X2': boxes[:, 2],
        'Y2': boxes[:, 3],
        'score': scores,
        'class': classes
    })
    f = StringIO()
    df.to_csv(f, index=False)
    f.seek(0)
    csv_dets = f.read()
    return csv_dets


def prepare_algorithm_error(exc_type, exc_value, exc_traceback):
    """
    prepare json-style results dict representing an algorithm error, as defined in the swagger
    """
    ex_type = exc_type.__name__
    if len(exc_value.args) > 0:
        ex_message = exc_value.args[0]
    else:
        ex_message = ''
    f = StringIO()
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
    f.seek(0)
    stack_trace = f.read()
    return ex_type, ex_message, stack_trace
