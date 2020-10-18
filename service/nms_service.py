from deep_wisdom.service.deepwisdom_handler_mock import REQUEST_EXAMPLE
from .detection_api_nms_performer import DetectionApiNmsPerformer
from .nms_request_handler import NmsRequestHandler


def main():
    nms_performer = DetectionApiNmsPerformer()
    nms_performer.create_session()
    nms_handler = NmsRequestHandler(nms_performer)
    nms_handler(REQUEST_EXAMPLE)
