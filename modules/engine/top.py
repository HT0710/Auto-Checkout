from collections import deque, defaultdict
from typing import Dict, List

import numpy as np
import cv2

from ..detection import ArucoDetector, ObjectDetector
from .engine import CameraEngine
from ..utils import load_config
from .server import Server


class TopEngine(CameraEngine):
    def __init__(self, camera_ids: List[int], engine_configs: Dict) -> None:
        super().__init__(
            camera_ids, engine_configs["camera"], engine_configs["calibration"]
        )
        self.aruco_detector = ArucoDetector(**load_config("configs/aruco.yaml"))
        self.object_detector = ObjectDetector(**engine_configs["detection"])
        self.detect_history = deque([], maxlen=10)

    def callback(self, image: np.ndarray) -> np.ndarray:
        process_image = image.copy()

        process_image = self.undistort(process_image)

        self.aruco_detector.draw(
            process_image, *self.aruco_detector.detect(process_image)
        )

        boxes = self.object_detector.detect(process_image)

        self.detect_history.append(len(boxes))

        cv2.putText(
            img=process_image,
            text=f"Count: {int(np.mean(self.detect_history))}",
            org=(20, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 0),
            thickness=2,
        )

        products = defaultdict(int)

        for box in boxes:
            # xyxy location
            x1, y1, x2, y2, _, idx = map(int, box)

            products[idx] += 1

            cv2.rectangle(
                img=process_image,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=(255, 255, 0),
                thickness=2,
            )

        Server.set(
            "products", str([{"code": k, "quantity": v} for k, v in products.items()])
        )

        return process_image

    def run(self) -> None:
        return super().run()
