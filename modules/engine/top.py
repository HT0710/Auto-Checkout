from collections import deque
from typing import Dict, List

import numpy as np
import cv2

from ..detection import ArucoDetector, ObjectDetector
from ..utils import load_config
from .camera import Camera


class TopEngine:
    def __init__(self, camera_ids: List[int], engine_configs: Dict) -> None:
        """
        Initializes a TopEngine object.

        Parameters
        ----------
        camera_ids : List[int]
            A list of integers representing the IDs of cameras associated with this engine.
        engine_configs : Dict
            A dictionary containing engine configurations.

        Notes
        -----
        This method initializes a TopEngine object with cameras, ArUco detector, object detector,
        and a deque for detecting history.
        """
        self.cameras = [
            Camera(device_id=value, **engine_configs["camera"]) for value in camera_ids
        ]
        self.aruco_detector = ArucoDetector(**load_config("configs/aruco.yaml"))
        self.object_detector = ObjectDetector(**engine_configs["detection"])
        self.detect_history = deque([], maxlen=10)

    def callback(self, image: np.ndarray) -> np.ndarray:
        """
        Performs callback operations on the image.

        Parameters
        ----------
        image : np.ndarray
            Input image to be processed.

        Returns
        -------
        np.ndarray
            Processed image.

        Notes
        -----
        This method performs ArUco marker detection, object detection, and draws
        bounding boxes and text on the image. It also calculates and displays
        the average count of detected objects.
        """

        # Create image clone
        process_image = image.copy()

        # Detect aruco
        self.aruco_detector.draw(
            process_image, *self.aruco_detector.detect(process_image)
        )

        # Detect object
        boxes = self.object_detector.detect(process_image)

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

        # Store result
        self.detect_history.append(len(boxes))

        result = round(np.mean(self.detect_history))

        # Add text to frame
        cv2.putText(
            img=process_image,
            text=f"Count: {result}",
            org=(20, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 0),
            thickness=2,
        )

        return {"frame": process_image, "results": result}
