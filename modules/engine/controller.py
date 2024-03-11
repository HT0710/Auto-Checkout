from typing import Dict, List, Tuple
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
                        if name == 0:
                            cv2.putText(
                                img=frame,
                                text=str("Unsure"),
                                org=point,
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                color=(0, 255, 255),
                                thickness=2,
                            )
                            continue

                        if name == -1:
                            cv2.putText(
                                img=frame,
                                text=str("Unknow"),
                                org=point,
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,
                                color=(0, 0, 255),
                                thickness=2,
                            )
                            continue

                        cv2.putText(
                            img=frame,
                            text=str(name),
                            org=point,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(255, 255, 0),
                            thickness=2,
                        )

            frames.append(frame)

        review = np.concatenate(frames, axis=1)

        # Display frame
        cv2.namedWindow("Review", cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow("Review", 1440, 480)
        cv2.imshow("Review", review)

        # Check for delay on each camera
        stop = not all([engine.delay() for engine in self.engines.values()])

        return stop, results

    def _calculate_distance(self, pa, pb):
        x1, y1, x2, y2 = *pa, *pb
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def _calculate_angle(self, ax, ay, bx, by):
        # Calculate the vectors representing the lines
        m1, n1 = ay[0] - ax[0], ay[1] - ax[1]
        m2, n2 = by[0] - bx[0], by[1] - bx[1]

        # Calculate the dot product and cross product of the vectors
        dot_product = m1 * m2 + n1 * n2
        cross_product = m1 * n2 - m2 * n1

        # Calculate the magnitude of the vectors
        magnitude1 = math.sqrt(m1**2 + n1**2)
        magnitude2 = math.sqrt(m2**2 + n2**2)

        # If either magnitude is zero, return a default value (e.g., 0)
        if magnitude1 * magnitude2 == 0:
            return 0

        # Calculate the angle between the lines in radians
        angle_rad = math.acos(dot_product / (magnitude1 * magnitude2))

        # Determine whether the angle is obtuse or acute based on the sign of the cross product
        if cross_product < 0:
            angle_rad = 2 * math.pi - angle_rad

        # Convert the angle to degrees
        angle_deg = angle_rad * 180 / math.pi

        return angle_deg

    def _calculate_coordinate(self, point, anchor_point: Tuple, dead_point: Tuple):
        distance = self._calculate_distance(pa=anchor_point, pb=point)

        angle = self._calculate_angle(
            ax=anchor_point, ay=point, bx=anchor_point, by=dead_point
        )

        return (distance, angle)

    def _coordinates_mapping(self, target, data, height, width):
        result = [-1] * len(target)

        anchor = target.pop(0)
        checker = [
            self._calculate_coordinate(
                point=current_point, anchor_point=anchor, dead_point=(0, anchor[1])
            )
            for current_point in target
        ]

        if len(target) < len(data):
            return result

        solved = 1
        for i, (point, (idx, conf)) in enumerate(data.items()):
            if i == 0:
                result[i] = self.classes[idx]
                anchor = point
                continue

            coord = self._calculate_coordinate(
                point=point, anchor_point=anchor, dead_point=(anchor[0], height)
            )

            candidates = [
                self._calculate_distance(pa=ground_truth, pb=coord)
                for ground_truth in checker
            ]

            best = min(candidates)

            for j in range(candidates.index(best) + 1):
                checker[j] = (-1, -1)

            result[candidates.index(best) + solved] = self.classes[idx]

        return result

    def run(self) -> None:
        Server.set("status", "SCAN")
        stop = False

        width, height = self.engines["top"].camera.size()

        # Main loop for capturing frames from cameras
        while not stop:
            signal = Server.get("status")

            stop, results = self._show(signal)

            if signal != "SCAN":
                continue

            top, left, right = results.values()

            matrix = [[-1] * top["total"] for _ in range(2)]

            if not top["total"]:
                continue

            matrix[0] = self._coordinates_mapping(
                target=top["left"].copy(), data=left, height=height, width=width
            )

            data = top["right"].copy()
            anchor = data.pop(0)
            checker = [
                self._calculate_coordinate(
                    point=current_point,
                    anchor_point=anchor,
                    dead_point=(width, anchor[1]),
                )
                for current_point in data
            ]

            if top["total"] >= len(right):
                solved = 1
                for i, (point, (idx, conf)) in enumerate(right.items()):
                    if i == 0:
                        matrix[1][i] = self.classes[idx]
                        anchor = point
                        continue

                    coord = self._calculate_coordinate(
                        point=point, anchor_point=anchor, dead_point=(anchor[0], height)
                    )

                    candidates = [
                        self._calculate_distance(pa=ground_truth, pb=coord)
                        for ground_truth in checker
                    ]

                    best = min(candidates)

                    for j in range(candidates.index(best) + 1):
                        checker[j] = (-1, -1)

                    matrix[1][candidates.index(best) + solved] = self.classes[idx]

            # print(matrix)
            print()

            self.products = {}

            for point, left, right in zip(top["left"], matrix[0], matrix[1][::-1]):
                if left == right or left == -1 or right == -1:
                    self.products[point] = left if left != -1 else right
                else:
                    self.products[point] = 0

            # print(self.products)

            continue

            # Server.set(
            #     "products",
            #     str([{"code": k, "quantity": v} for v in self.products.values()]),
            # )
            # Server.set("message", "Detect successfully with x% certainty.")

        # Release camera resources
        [engine.release() for engine in self.engines.values()]
