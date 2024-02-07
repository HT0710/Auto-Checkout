from typing import Dict, List

from rich import print
import numpy as np

from .camera import Camera


class CameraEngine:
    def __init__(self, camera_ids: List[int], camera_configs: Dict) -> None:
        """
        Initializes a CameraEngine object.

        Parameters
        ----------
        camera_ids : List[int]
            List of camera IDs to be used.
        camera_configs : Dict
            Configuration dictionary for camera setup.
        """
        self.cameras = [
            Camera(device_id=cam_id, **camera_configs) for cam_id in camera_ids
        ]

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
