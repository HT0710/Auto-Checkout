from typing import Dict, Union

import numpy as np
import cv2

from ..detection import ObjectDetector
from ..detection import ArucoDetector
from ..utils import load_config
from .camera import Camera


class Engine:
    def __init__(self, index: int, engine_configs: Dict) -> None:
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
        self.camera = Camera(device_id=index, **engine_configs["camera"])
        self.object_detector = ObjectDetector(**engine_configs["detection"])
        self.current_frame = iter(self.camera)

    def get_frame(self) -> np.ndarray:
        return next(self.current_frame)

    def process(self, image: np.ndarray) -> Union[np.ndarray, Dict]:
        raise NotImplementedError()

    def delay(self) -> bool:
        return self.camera.delay()

    def release(self) -> None:
        self.camera.release()


class TopEngine(Engine):
    def __init__(self, index: int, engine_configs: Dict) -> None:
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
        super().__init__(index, engine_configs)
        self.aruco_detector = ArucoDetector(**load_config("configs/aruco.yaml"))

    def process(self, image: np.ndarray) -> Union[np.ndarray, Dict]:
        """
        Processing operations on the image.

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

        products = {}

        center_points = []

        for box in boxes:
            # xyxy location
            x1, y1, x2, y2, _, idx = map(int, box)

            conf = round(box[4], 2)

            cv2.rectangle(
                img=process_image,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=(255, 255, 0),
                thickness=2,
            )

            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            cv2.circle(process_image, center, 8, (255, 0, 255), -1)

            center_points.append(center)

            products[center] = conf

        # Add text to frame
        # cv2.putText(
        #     img=process_image,
        #     text=f"Count: {len(center_points)}",
        #     org=(20, 50),
        #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #     fontScale=2,
        #     color=(255, 255, 0),
        #     thickness=3,
        # )

        cv2.putText(
            img=process_image,
            text="Top",
            org=(20, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(0, 0, 255),
            thickness=5,
        )

        sorted_points_left = sorted(center_points, key=lambda p: (p[0], p[1]))

        # for i, center in enumerate(sorted_points_left):
        #     cv2.putText(
        #         img=process_image,
        #         text=f"{i}",
        #         org=(center[0] - 50, center[1]),
        #         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #         fontScale=2,
        #         color=(0, 255, 0),
        #         thickness=4,
        #     )

        sorted_points_right = sorted(center_points, key=lambda p: (-p[0], p[1]))

        # for i, center in enumerate(sorted_points_right):
        #     cv2.putText(
        #         img=process_image,
        #         text=f"{i}",
        #         org=(center[0] + 10, center[1]),
        #         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #         fontScale=2,
        #         color=(0, 0, 255),
        #         thickness=4,
        #     )

        sorted_products_left = {
            key: products[key]
            for key in sorted(products, key=lambda x: sorted_points_left.index(x))
        }

        sorted_products_right = {
            key: products[key]
            for key in sorted(products, key=lambda x: sorted_points_right.index(x))
        }

        return process_image, {
            "left": sorted_products_left,
            "right": sorted_products_right,
        }


class SideEngine(Engine):
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Processing operations on the image.

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
        products = {}

        for box in boxes:
            # xyxy location
            x1, y1, x2, y2, _, idx = map(int, box)

            conf = round(box[4], 2)

            center = ((x1 + x2) // 2, y2)

            # cv2.circle(process_image, center, 8, (255, 0, 255), -1)

            products[center] = (idx, conf)

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

        return process_image, products


class LeftEngine(SideEngine):
    def process(self, image: np.ndarray) -> np.ndarray:
        processed_image, products = self.preprocess(image)

        cv2.putText(
            img=processed_image,
            text="Left",
            org=(20, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(255, 255, 0),
            thickness=5,
        )

        sorted_products = dict(
            sorted(products.items(), key=lambda x: (-x[0][1], x[0][0]))
        )

        for i, center in enumerate(sorted_products.keys()):
            cv2.putText(
                img=processed_image,
                text=f"{i}",
                org=center,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 255, 0),
                thickness=4,
            )

        return processed_image, sorted_products


class RightEngine(SideEngine):
    def process(self, image: np.ndarray) -> np.ndarray:
        processed_image, products = self.preprocess(image)

        cv2.putText(
            img=processed_image,
            text="Right",
            org=(20, 50),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=(0, 255, 255),
            thickness=5,
        )

        sorted_products = dict(
            sorted(products.items(), key=lambda x: (-x[0][1], -x[0][0]))
        )

        for i, center in enumerate(sorted_products.keys()):
            cv2.putText(
                img=processed_image,
                text=f"{i}",
                org=center,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2,
                color=(0, 0, 255),
                thickness=4,
            )

        return processed_image, sorted_products
