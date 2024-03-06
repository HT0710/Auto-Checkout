from typing import Dict, List

import numpy as np

from .model import ExtractLabeling, YoloLabeling
from ..utils import device_handler


class AutoLabeling:
    def __init__(self, method: str, device: str, extract: Dict, yolo: Dict) -> None:
        """
        Initialize AutoLabeling instance.

        Parameters
        ----------
        method : str
            Method for labeling, either "extract" or "yolo".
        device : str
            Device type for processing.
        extract : Dict
            Parameters for extract labeling.
        yolo : Dict
            Parameters for YOLO labeling.
        """
        self.method = self._check_method(method)
        self.device = device_handler(device)

        if self.method == "extract":
            self.labeler = ExtractLabeling(**extract, device=self.device)

        if self.method == "yolo":
            self.labeler = YoloLabeling(**yolo, device=self.device)

    def _check_method(self, value: str) -> str:
        """
        Check if the method is valid.

        Parameters
        ----------
        value : str
            The method to check.

        Returns
        -------
        str
            The validated method.

        Raises
        ------
        ValueError
            If method is not 'extract' or 'yolo'.
        """
        value = str(value).strip().lower()

        if value not in ["extract", "yolo"]:
            raise ValueError("Invalid method. Available options: extract, yolo")

        return value

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
            The bounding box coordinates.
        """
        return self.labeler.get_bounding_box(image)
