from deep_wisdom.service.deepwisdom_handler_mock import REQUEST_EXAMPLE
from .detection_api_nms_performer import DetectionApiNmsPerformer
from .nms_api import NmsApi
from .nms_handler import NmsHandler


def main():
    nms_performer = DetectionApiNmsPerformer()
    nms_performer.create_session()
    nms_api = NmsApi()
    nms_handler = NmsHandler(nms_performer, nms_api)
    nms_handler(REQUEST_EXAMPLE)
