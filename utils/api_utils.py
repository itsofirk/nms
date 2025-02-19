import traceback

import numpy as np
import pandas as pd


def bbox2coords(bbox):
    """
    expands [left,top,right,bottom] to a 5-point polygon in [x_col,y_col] format (counter clockwise)
    """
    left, top, right, bottom = np.squeeze(bbox)
    y = [top, bottom, bottom, top, top]
    x = [left, left, right, right, left]
    return np.vstack((x, y)).T


def coords2bbox(coords):
    """ calculates [left,top,right,bottom] from [x_col,y_col] """
    return np.hstack((coords.min(axis=0), coords.max(axis=0)))


def pack_detections(boxes, scores, classes):
    """
    structure detections into json-style format as defined in the swagger file
    """
    detections = []
    for box, score, class_name in zip(boxes, scores, classes):
        coords = bbox2coords(box)
        outer_ring = coords.tolist()
        pixel_polygon = {
            'type': 'Polygon',
            'coordinates': [outer_ring]
        }
        detection = {
            'classification': class_name,
            'grade': float(score),
            'pixelLocation': pixel_polygon
        }
        detections.append(detection)
    return detections


def unpack_detections(detections):
    """
    parse detections from the json-style format define in the swagger file into boxes, scores, classes
    """
    coords = []
    scores = []
    classes = []

    for dtc in detections:
        coords.append(dtc['pixelLocation']['coordinates'][0])
        scores.append(dtc['grade'])
        classes.append(dtc['classification'])
    coords = np.array(coords)
    scores = np.array(scores).astype("float32")
    classes = np.array(classes)

    boxes = np.array([coords2bbox(x) for x in coords])
    check_correspondence(boxes, scores, classes)
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
    return df.to_csv(index=False)


def parse_exception(exception):
    """
    , exc_type, exc_value, exc_traceback
    prepare json-style results dict representing an algorithm error, as defined in the swagger
    :param exception
    :_param exc_type, exc_value, exc_traceback: the results of calling sys.exc_info() after catching the exception
    :return: results - a json-like object (composed of dicts and lists) representing the error.
    """
    if not exception:
        return
    exc_value = ''
    if len(exception.args) > 0:
        exc_value = exception.args[0]
    stacktrace = traceback.format_exception(type(exception), exception, exception.__traceback__)
    return type(exception).__name__, exc_value, stacktrace


# NmsAPI

def prepare_results(boxes, scores, classes):
    """
    prepare detection results. usually some sort of textual serialization.
    :param boxes: [Nx4] np.array of boxes: x_min, y_min, x_max, y_max
    :param scores: [N] np.array of sorted scores (highest first)
    :param classes: [N] np.array of string class names
    :return: results - a json-like object (composed of dicts and lists) of detection results.
    """
    return {
        'detections': pack_detections(boxes, scores, classes),
        'statusType': 'stripDetections'
    }


def check_correspondence(boxes, scores, classes):
    if not len(boxes) == len(scores) == len(classes):
        raise ValueError("One or more of these lists {boxes, scores, classes} are not equal in size")
    return True
