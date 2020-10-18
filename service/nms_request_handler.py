from utils import api_utils


class NmsRequestHandler:

    def __init__(self, nms_performer):
        self.nms_performer = nms_performer

    def __call__(self, request):
        return self.handle_request(request)

    def handle_request(self, request):
        boxes, scores, classes, input_metadata = api_utils.parse_request(request)
        try:
            boxes, scores, classes = self.nms_performer.suppress(boxes, scores, classes, input_metadata)
            results = api_utils.prepare_results(boxes, scores, classes)
        except Exception as e:
            exc_type, exc_value, stacktrace = api_utils.prepare_algorithm_error(e)
            results = {'error': 'ERROR! See Stack Trace', 'statusType': 'algorithmError',
                       'type': exc_type, 'message': exc_value, 'stack_trace': stacktrace}
            print(f"EXCEPTION: {results}")
        return results
