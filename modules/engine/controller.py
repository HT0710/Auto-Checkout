from typing import Dict, Tuple
import math

import numpy as np
import cv2

from .engine import TopEngine, LeftEngine, RightEngine
from ..utils import load_config
from .server import Server


class Controller:
    def __init__(self, top_id: int, left_id: int, right_id: int) -> None:
        """
        _summary_

        Parameters
        ----------
        top_id
            _description_
        left_id
            _description_
        right_id
            _description_
        """

        # Define engine initialization
        _engines_initialize = [
            ("top", TopEngine, top_id, "configs/engine/top.yaml"),
            ("left", LeftEngine, left_id, "configs/engine/side.yaml"),
            ("right", RightEngine, right_id, "configs/engine/side.yaml"),
        ]

        self.engines = {
            name: engine(index=idx, engine_configs=load_config(config))
            for name, engine, idx, config in _engines_initialize
        }

        self.classes = self.engines["left"].object_detector.classes

    def _show(self, signal: str) -> Tuple[bool, Dict]:
        results = {}
        frames = []

        for name, engine in self.engines.items():
            frame = engine.get_frame()

            # Perform callback on the engine
            if signal == "SCAN":
                frame, results[name] = engine.process(frame)

            if name == "top":
                if hasattr(self, "products"):
                    for point, name in self.products.items():
                        cv2.putText(
                            img=frame,
                            text=str(name),
                            org=point,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(255, 255, 0),
                            thickness=2,
                        )

            frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)

            frames.append(frame)

        review = np.concatenate(frames, axis=1)

        # Display frame
        cv2.namedWindow("Review", cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("Review", review)

        # Check for delay on each camera
        stop = (
            True
            if not all([engine.delay() for engine in self.engines.values()])
            else False
        )

        return stop, results

    def _min_max_normalize(self, data):
        min_val = min(data)
        max_val = max(data)

        if min_val == max_val:
            return [0] * len(data)

        normalized_data = [round((x - min_val) / (max_val - min_val), 2) for x in data]

        return normalized_data

    def _calculate_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def _calculate_angle(ax, ay, bx, by):
        # Calculate the vectors representing the lines
        m1, n1 = ay[0] - ax[0], ay[1] - ax[1]
        m2, n2 = by[0] - bx[0], by[1] - bx[1]

        # Calculate the dot product of the vectors
        dot_product = m1 * m2 + n1 * n2

        # Calculate the magnitude of the vectors
        magnitude1 = math.sqrt(m1**2 + n1**2)
        magnitude2 = math.sqrt(m2**2 + n2**2)

        # Calculate the angle between the lines in radians
        angle_rad = math.acos(dot_product / (magnitude1 * magnitude2))

        # Convert the angle to degrees
        angle_deg = angle_rad * 180 / math.pi

        return angle_deg

    def run(self) -> None:
        stop = False

        # Main loop for capturing frames from cameras
        while not stop:
            signal = Server.get("status")

            stop, results = self._show(signal)

            if signal != "SCAN":
                continue

            top, left, right = results.values()

            checker = list(top["left"].keys())

            self.products = {}

            for (l_idx, l_conf), (r_idx, r_conf) in zip(left.values(), right.values()):
                if not checker:
                    break

                self.products[checker.pop(0)] = self.classes[l_idx]

                if not checker:
                    break

                self.products[checker.pop(-1)] = self.classes[r_idx]

            # Server.set(
            #     "products",
            #     str([{"code": k, "quantity": v} for v in self.products.values()]),
            # )
            # Server.set("message", "Detect successfully with x% certainty.")

        # Release camera resources
        [engine.release() for engine in self.engines.values()]
