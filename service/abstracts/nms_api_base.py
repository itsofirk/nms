import abc


class NmsApiBase:
    @abc.abstractmethod
    def parse_request(self, request):
        """
        parse request dictionary. return parsed detections and additional metadata.
        :param request: request dictionary
        :return: boxes - [Nx4] np.array of boxes: x_min, y_min, x_max, y_max
                 scores - [N] np.array of sorted scores (highest first)
                 classes - [N] np.array of string class names
                 input_metadata - a dictionary of additional input params. can be empty.
        """
        boxes, scores, classes, input_metadata = None, None, None, None
        return boxes, scores, classes, input_metadata

    @abc.abstractmethod
    def prepare_results(self, boxes, scores, classes):
        """
        prepare detection results. usually some sort of textual serialization.
        :param boxes: [Nx4] np.array of boxes: x_min, y_min, x_max, y_max
        :param scores: [N] np.array of sorted scores (highest first)
        :param classes: [N] np.array of string class names
        :return: results - a json-like object (composed of dicts and lists) of detection results.
        """
        results = None
        return results

    @abc.abstractmethod
    def prepare_error_results(self, exc_type, exc_value, exc_traceback):
        """
        prepare results dict in case an exception was raised
        :param exc_type, exc_value, exc_traceback: the results of calling sys.exc_info() after catching the exception
        :return: results - a json-like object (composed of dicts and lists) representing the error.
        """
        results = None
        return results
