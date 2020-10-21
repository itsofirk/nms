from utils import api_utils
from logic.cpu_nms import multiclass_non_maximum_suppression as multiclass_nms


def new_handle_request(request):
    boxes, scores, classes, input_metadata = api_utils.parse_request(request)
    try:
        boxes, scores, classes = multiclass_nms(boxes, scores, classes, input_metadata)
        results = api_utils.prepare_results(boxes, scores, classes)
    except Exception as e:
        exc_type, exc_value, stacktrace = api_utils.parse_exception(e)
        results = {'error': 'ERROR! See Stack Trace', 'statusType': 'algorithmError',
                   'type': exc_type, 'message': exc_value, 'stack_trace': stacktrace}
        print(f"EXCEPTION: {results}")
    return results


class NmsRequestHandler:

    def __init__(self, nms_performer):
        self.nms_performer = nms_performer

    def __call__(self, request):
        return new_handle_request(request)

    def handle_request(self, request):
        boxes, scores, classes, input_metadata = api_utils.parse_request(request)
        try:
            boxes, scores, classes = self.nms_performer.suppress(boxes, scores, classes, input_metadata)
            results = api_utils.prepare_results(boxes, scores, classes)
        except Exception as e:
            exc_type, exc_value, stacktrace = api_utils.parse_exception(e)
            results = {'error': 'ERROR! See Stack Trace', 'statusType': 'algorithmError',
                       'type': exc_type, 'message': exc_value, 'stack_trace': stacktrace}
            print(f"EXCEPTION: {results}")
        return results
