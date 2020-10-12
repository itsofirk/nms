from .abstracts import NmsApiBase
from .utils import api_utils


class NmsApi(NmsApiBase):
    def parse_request(self, request):
        """ see NmsApiBase """
        detections = request.pop('detections')
        boxes, scores, classes = api_utils.parse_detections_json(detections)
        input_metadata = request
        return boxes, scores, classes, input_metadata

    def prepare_results(self, boxes, scores, classes):
        """ see NmsApiBase """
        result = {'detections': api_utils.prepare_detections_json(boxes, scores, classes),
                  'statusType': 'stripDetections'}
        return result

    def prepare_error_results(self, exc_type, exc_value, exc_traceback):
        """ see NmsApiBase """
        return api_utils.prepare_algorithm_error(exc_type, exc_value, exc_traceback)
