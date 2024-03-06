from typing import List, Tuple

import numpy as np

from ...detection.yolov8 import ObjectDetector


class YoloLabeling(ObjectDetector):
    def _xyxy_to_xywh(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[int]:
        """
        Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format.

        Parameters
        ----------
        x1 : int
            x-coordinate of the top-left corner.
        y1 : int
            y-coordinate of the top-left corner.
        x2 : int
            x-coordinate of the bottom-right corner.
        y2 : int
            y-coordinate of the bottom-right corner.

        Returns
        -------
        tuple
            (x, y, width, height)
        """
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        return x, y, width, height

    def get_bounding_box(self, image: np.ndarray) -> List[int]:
        """
        Get bounding box coordinates for the given image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        List[int]
            List containing bounding box coordinates in (x, y, width, height) format.
        """
        return [*self._xyxy_to_xywh(*map(int, self.detect(image)[0][:4]))]
