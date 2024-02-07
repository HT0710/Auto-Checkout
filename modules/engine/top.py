from typing import Dict, List

import numpy as np
import cv2

from ..detection import ArucoDetector, ObjectDetector
from .engine import CameraEngine
from ..utils import load_config


class TopEngine(CameraEngine):
    def __init__(self, camera_ids: List[int], camera_configs: Dict) -> None:
        super().__init__(camera_ids, camera_configs)
        self.aruco_detector = ArucoDetector(**load_config("configs/aruco.yaml"))
        self.object_detector = ObjectDetector(**load_config("configs/yolov8.yaml"))

    def callback(self, image: np.ndarray) -> np.ndarray:
        process_image = image.copy()

        self.aruco_detector.draw(
            process_image, *self.aruco_detector.detect(process_image)
        )

        boxes = self.object_detector.detect(process_image)

        cv2.putText(
            img=process_image,
            text=f"Count: {len(boxes)}",
            org=(20, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 0),
            thickness=2,
        )

        for box in boxes:
            # xyxy location
            x1, y1, x2, y2 = map(int, box[:4])

            cv2.rectangle(
                img=process_image,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=(255, 255, 0),
                thickness=2,
            )

        return process_image

    def run(self) -> None:
        return super().run()
