from utils import api_utils
from logic.cpu_nms import multiclass_non_maximum_suppression as multiclass_nms


def handle_request(request):
    boxes, scores, classes = api_utils.unpack_detections(request['detections'])
    try:
        boxes, scores, classes = multiclass_nms(boxes, scores, classes, request)
        results = api_utils.prepare_results(boxes, scores, classes)
    except Exception as e:
        exc_type, exc_value, stacktrace = api_utils.parse_exception(e)
        results = {
            'error': 'ERROR! See Stack Trace',
            'statusType': 'algorithmError',
            'type': exc_type,
            'message': exc_value,
            'stack_trace': stacktrace
        }
        print(f"EXCEPTION: {results}")
    return results
