from .nms_request_handler import handle_request


def main():
    # TODO: add queue manager implementation
    # wait for messages:
    # then handle_request(message)
    raise NotImplemented()


def test():
    assert RESULT_EXAMPLE == handle_request(REQUEST_EXAMPLE)


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
        }, {
            "grade": 0.9,
            "pixelLocation": {
                "type": "Polygon",
                "coordinates": [
                    [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]
                ]
            },
            "classification": "rectangle"
        },
        {
            "grade": 0.7,
            "pixelLocation": {
                "type": "Polygon",
                "coordinates": [
                    [[1, 1], [0, 11], [11, 11], [11, 0], [1, 1]]
                ]
            },
            "classification": "rectangle"
        }, {
            "grade": 0.5,
            "pixelLocation": {
                "type": "Polygon",
                "coordinates": [
                    [[1, 1], [0, 10], [11, 11], [10, 0], [1, 1]]
                ]
            },
            "classification": "not_rectangle"
        }
    ],
    "num_positive_detections": {
        "class_name": 251
    },
    "statusType": "stripDetections"
}
RESULT_EXAMPLE = {
    'detections': [
        {
            'classification': 'class_name',
            'grade': 0.8105800151824951,
            'pixelLocation': {
                'type': 'Polygon',
                'coordinates': [
                    [
                        [4426.42724609375, 33574.12890625],
                        [4426.42724609375, 33651.75],
                        [4463.306640625, 33651.75],
                        [4463.306640625, 33574.12890625],
                        [4426.42724609375, 33574.12890625]
                    ]
                ]
            }
        }, {
            'classification': 'class_name',
            'grade': 0.5132514834403992,
            'pixelLocation': {
                'type': 'Polygon',
                'coordinates': [
                    [
                        [4176.28564453125, 39387.12890625],
                        [4176.28564453125, 39472.26953125],
                        [4214.9267578125, 39472.26953125],
                        [4214.9267578125, 39387.12890625],
                        [4176.28564453125, 39387.12890625]
                    ]
                ]
            }
        },
        {
            'classification': 'not_rectangle',
            'grade': 0.5,
            'pixelLocation': {
                'type': 'Polygon',
                'coordinates': [
                    [
                        [0.0, 0.0],
                        [0.0, 11.0],
                        [11.0, 11.0],
                        [11.0, 0.0],
                        [0.0,
                         0.0]]
                ]
            }
        },
        {
            'classification': 'rectangle',
            'grade': 0.699999988079071,
            'pixelLocation': {
                'type': 'Polygon',
                'coordinates': [
                    [
                        [0.0, 0.0],
                        [0.0, 10.0],
                        [10.0, 10.0],
                        [10.0, 0.0],
                        [0.0, 0.0]
                    ]
                ]
            }
        }
    ],
    'statusType': 'stripDetections'
}
