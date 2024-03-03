from typing import List

import cv2

from ..utils import load_config
from .side import SideEngine
from .top import TopEngine


class CameraControler:
    def __init__(self, top_ids: List[int] = None, side_ids: List[int] = None) -> None:
        """
        Initializes a CameraEngine object.

        Parameters
        ----------
        top_ids : List[int], optional
            A list of integers representing the IDs of top cameras, by default None.
        side_ids : List[int], optional
            A list of integers representing the IDs of side cameras, by default None.
        """
        self.engines = []

        # Define camera configurations
        camera_configs = [
            (top_ids, TopEngine, "configs/engine/top.yaml"),
            (side_ids, SideEngine, "configs/engine/side.yaml"),
        ]

        # Iterate over camera configurations and instantiate engines
        for camera_ids, engine_class, config_file in camera_configs:
            # If camera IDs are not provided, skip instantiation
            if not camera_ids:
                continue

            # Instantiate engine
            self.engines.append(
                engine_class(
                    camera_ids=camera_ids,
                    engine_configs=load_config(config_file),
                )
            )

    def run(self) -> None:
        """
        Runs the camera capture loop for all cameras.

        Notes
        -----
        This method iterates over all engines and their associated cameras, capturing frames
        from each camera and displaying them using OpenCV. It continues until a camera's delay
        limit is reached.

        After the loop exits, it releases all camera resources.
        """

        controler = {}

        # Initialize controller dictionary with cameras for each engine
        for engine in self.engines:
            controler[engine] = [
                {"self": camera, "current": iter(camera)} for camera in engine.cameras
            ]

        stop = False

        # Main loop for capturing frames from cameras
        while not stop:
            for engine, cameras in controler.items():
                for camera in cameras:
                    frame = next(camera["current"])

                    # Perform callback on the engine
                    frame = engine.callback(frame)

                    # Display frame
                    cv2.imshow(str(camera["self"].device_id), frame)

                    # Check for delay on each camera
                    if not camera["self"].delay(camera["self"].wait):
                        stop = True

        # Release camera resources
        [
            camera["self"].release()
            for cameras in controler.values()
            for camera in cameras
        ]
