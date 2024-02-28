from typing import Dict, List

from rich import print
import numpy as np
import cv2

from .camera import Camera


class CameraEngine:
    def __init__(
        self, camera_ids: List[int], camera_configs: Dict, calibration: Dict
    ) -> None:
        """
        Initializes a CameraEngine object.

        Parameters
        ----------
        camera_ids : List[int]
            List of camera IDs to be used.
        camera_configs : Dict
            Configuration dictionary for camera setup.
        calibration : Dict
            Configuration dictionary for camera calibration.
        """
        self.cameras = [
            Camera(device_id=cam_id, **camera_configs) for cam_id in camera_ids
        ]
        if calibration:
            self.camera_params = np.load(calibration["path"])

    def undistort(self, image: np.ndarray) -> np.ndarray:
        if not hasattr(self, "camera_params"):
            return image

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
        """
        Callback method to be overridden by subclasses.

        Parameters
        ----------
        image : np.ndarray
            Input image.

        Returns
        -------
        np.ndarray
            Processed image.
        """
        print("[yellow][WARNING] Callback method needs to be overridden.[/]", end="\r")
        return image

    def run(self) -> None:
        """Runs the camera capture loop for all cameras."""
        [camera.run(callback=self.callback) for camera in self.cameras]
