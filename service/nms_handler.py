import sys

class NmsHandler:

    def __init__(self, nms_performer, nms_api):
        self.nms_performer = nms_performer
        self.nms_api = nms_api

    def __call__(self, request):
        return self.handle_request(request)

    def handle_request(self, request):
        boxes, scores, classes, input_metadata = self.nms_api.parse_request(request)
        try:
            boxes, scores, classes = self.nms_performer.suppress(boxes, scores, classes, input_metadata)
            results = self.nms_api.prepare_results(boxes, scores, classes)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print("EXCEPTION: {}".format(e))
            ex_type, ex_message, stack_trace = self.nms_api.prepare_error_results(exc_type, exc_value, exc_traceback)
            results = {'error': 'ERROR! See Stack Trace', 'statusType': 'algorithmError',
                       'type': ex_type, 'message': ex_message, 'stack_trace': stack_trace}
        return results
