from collections import defaultdict
from typing import List

import numpy as np
import cv2

from ..utils import load_config
from .side import SideEngine
from .top import TopEngine
from .server import Server


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
        self.engines = {}

        # Define camera configurations
        camera_configs = [
            (top_ids, "top", TopEngine, "configs/engine/top.yaml"),
            (side_ids, "side", SideEngine, "configs/engine/side.yaml"),
        ]

        # Iterate over camera configurations and instantiate engines
        for camera_ids, name, engine_class, config_file in camera_configs:
            # If camera IDs are not provided, skip instantiation
            if not camera_ids:
                continue

            # Create engine
            engine = engine_class(
                camera_ids=camera_ids,
                engine_configs=load_config(config_file),
            )

            # Instantiate engine
            self.engines[name] = {
                "self": engine,
                "cameras": [
                    {"self": camera, "current": iter(camera)}
                    for camera in engine.cameras
                ],
            }

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

        Server.set("status", "STOP")

        stop = False

        # Main loop for capturing frames from cameras
        while not stop:
            signal = Server.get("status")

            results = {"top": None, "side": []}

            frames = []

            for key, value in self.engines.items():
                engine = value["self"]

                for camera in value["cameras"]:
                    frame = next(camera["current"])

                    # Perform callback on the engine
                    if signal == "SCAN":
                        output = engine.callback(frame)

                        frame = output["frame"]

                        if key == "top":
                            results["top"] = output["results"]

                        if key == "side":
                            results["side"].append(output["results"])

                    frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)

                    frames.append(frame)

            review = np.concatenate(frames)

            # Display frame
            cv2.namedWindow("Review", cv2.WINDOW_GUI_NORMAL)
            cv2.imshow("Review", review)

            # Check for delay on each camera
            if not all(
                [
                    camera["self"].delay(camera["self"].wait)
                    for engine in self.engines.values()
                    for camera in engine["cameras"]
                ]
            ):
                stop = True

            if signal != "SCAN":
                continue

            total = results["top"]

            left, right = results["side"]
            conflict = right.copy()
            products = defaultdict(int)

            for key, value in left.items():
                if key in right:
                    if len(value) == len(right[key]):
                        products[key] += 1
                        conflict.pop(key)

                else:
                    conflict[key] = value

            for key in conflict.keys():
                products[key] += 1

            if len(products) > total:
                for key, value in products.items():
                    for conf in value:
                        if conf < 0.7:
                            products.pop(key)

            elif len(products) < total:
                Server.set(
                    "message",
                    str({"code": "0", "detail": "Some products are undetectable."}),
                )

            Server.set(
                "products",
                str([{"code": k, "quantity": v} for k, v in products.items()]),
            )
            # Server.set("message", "Detect successfully with x% certainty.")

        # Release camera resources
        [
            camera["self"].release()
            for engine in self.engines.values()
            for camera in engine["cameras"]
        ]
