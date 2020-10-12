from detection_api_nms_performer import DetectionApiNmsPerformer
from nms_api import NmsApi
from nms_handler import NmsHandler


def main():
    nms_performer = DetectionApiNmsPerformer()
    nms_api = NmsApi()
    nms_handler = NmsHandler(nms_performer, nms_api)


if __name__ == '__main__':
    main()
