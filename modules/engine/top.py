from collections import deque
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
        self.camera_params = np.load("weights/calibration_params.npz")
        self.detect_history = deque([], maxlen=10)

    def undistort(self, image: np.ndarray):
        mtx = self.camera_params["mtx"]
        dist = self.camera_params["dist"]

        h, w = image.shape[:2]

        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1)

        map_x, map_y = cv2.initUndistortRectifyMap(
            mtx, dist, None, new_mtx, (w, h), cv2.CV_32FC1
        )

        image = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC)

        x, y, w, h = roi

        return image[y : y + h, x : x + w]

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
