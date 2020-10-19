from logic.detection_api_nms_performer import DetectionApiNmsPerformer
from .nms_request_handler import NmsRequestHandler

REQUEST_EXAMPLE = {
    "detections": [
        {
            "grade": 0.7105799913406372,
            "pixelLocation": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [
                            4426.42724609375,
                            33574.12890625
                        ],
                        [
                            4426.42724609375,
                            33651.75
                        ],
                        [
                            4463.306640625,
                            33651.75
                        ],
                        [
                            4463.306640625,
                            33574.12890625
                        ],
                        [
                            4426.42724609375,
                            33574.12890625
                        ]
                    ]
                ]
            },
            "classification": "class_name"
        },
        {
            "grade": 0.8105799913406372,
            "pixelLocation": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [
                            4426.42724609375,
                            33574.12890625
                        ],
                        [
                            4426.42724609375,
                            33651.75
                        ],
                        [
                            4463.306640625,
                            33651.75
                        ],
                        [
                            4463.306640625,
                            33574.12890625
                        ],
                        [
                            4426.42724609375,
                            33574.12890625
                        ]
                    ]
                ]
            },
            "classification": "class_name"
        },
        {
            "grade": 0.5132514834403992,
            "pixelLocation": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [
                            4176.28564453125,
                            39387.12890625
                        ],
                        [
                            4176.28564453125,
                            39472.26953125
                        ],
                        [
                            4214.9267578125,
                            39472.26953125
                        ],
                        [
                            4214.9267578125,
                            39387.12890625
                        ],
                        [
                            4176.28564453125,
                            39387.12890625
                        ]
                    ]
                ]
            },
            "classification": "class_name"
        }
    ],
    "num_positive_detections": {
        "class_name": 251
    },
    "statusType": "stripDetections"
}
RESULT_EXAMPLE = {'detections': [{'grade': 0.8105800151824951, 'pixelLocation': {'type': 'Polygon', 'coordinates': [
    [[4426.42724609375, 33574.12890625], [4426.42724609375, 33651.75], [4463.306640625, 33651.75],
     [4463.306640625, 33574.12890625], [4426.42724609375, 33574.12890625]]]}, 'classification': 'class_name'},
                                 {'grade': 0.5132514834403992, 'pixelLocation': {'type': 'Polygon', 'coordinates': [
                                     [[4176.28564453125, 39387.12890625], [4176.28564453125, 39472.26953125],
                                      [4214.9267578125, 39472.26953125], [4214.9267578125, 39387.12890625],
                                      [4176.28564453125, 39387.12890625]]]}, 'classification': 'class_name'}],
                  'statusType': 'stripDetections'}


def main():
    # TODO: add queue manager implementation
    nms_performer = DetectionApiNmsPerformer()
    nms_performer.create_session()
    nms_handler = NmsRequestHandler(nms_performer)
    nms_handler(REQUEST_EXAMPLE)


def test():
    nms_performer = DetectionApiNmsPerformer()
    nms_performer.create_session()
    nms_handler = NmsRequestHandler(nms_performer)
    assert RESULT_EXAMPLE == nms_handler(REQUEST_EXAMPLE)
