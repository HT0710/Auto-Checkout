from typing import Dict, List

import numpy as np
import cv2

from ..detection import ObjectDetector
from .camera import Camera


class SideEngine:
    def __init__(self, camera_ids: List[int], engine_configs: Dict) -> None:
        """
        Initializes a SideEngine object.

        Parameters
        ----------
        camera_ids : List[int]
            A list of integers representing the IDs of cameras associated with this engine.
        engine_configs : Dict
            A dictionary containing engine configurations.

        Notes
        -----
        This method initializes a SideEngine object with cameras and an object detector.
        """
        self.cameras = [
            Camera(device_id=value, **engine_configs["camera"]) for value in camera_ids
        ]
        self.object_detector = ObjectDetector(**engine_configs["detection"])

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
        This method performs object detection and draws bounding boxes around detected objects.
        """

        # Create image clone
        process_image = image.copy()

        # Detect obejct
        boxes = self.object_detector.detect(process_image)

        # Products tracking
        products = []

        for box in boxes:
            # xyxy location
            x1, y1, x2, y2 = map(int, box[:4])

            conf, idx = round(box[4], 2), int(box[5])

            products.append({idx: conf})

            cv2.rectangle(
                img=process_image,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=(255, 255, 0),
                thickness=2,
            )

            cv2.putText(
                img=process_image,
                text=self.object_detector.classes[idx],
                org=(x1 + 10, y1 + 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 255, 0),
                thickness=2,
            )

        return {"frame": process_image, "results": products}
