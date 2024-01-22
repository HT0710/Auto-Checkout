from typing import List

from rembg import remove, new_session
import numpy as np
import cv2

from .utils import device_handler


class AutoLabeling:
    def __init__(
        self, model: str = "silueta", device: str = "auto", tensorrt: bool = False
    ) -> None:
        """
        Initializes an instance of the AutoLabeling class.

        Args:
            model (str, optional): The name of the model to be used. Defaults to "silueta".
            device (str): The device specification. Valid options: ["auto", "cpu", "cuda"]. Default to "auto".
            tensorrt (bool): Using TensorRT to speedup inference speed (CUDA only). Default to False.

        For more models information: https://github.com/danielgatis/rembg?tab=readme-ov-file#models
        """

        # Check device
        self._device = device_handler(device)

        # Initializes CPU provider
        _providers = ["CPUExecutionProvider"]

        if self._device == "cuda":
            # Add CUDA provider
            _providers.insert(0, "CUDAExecutionProvider")

            if tensorrt:
                # Add TensorRT provider
                _providers.insert(0, "TensorrtExecutionProvider")

        # Create new onnx session
        self._session = new_session(model, providers=_providers)

        # Define default contour threshold
        self._contour_threshold = 20

    def _remove_background(self, image: np.ndarray) -> np.ndarray:
        """
        Removes the background from the input image using the initialized model.

        Args:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The image with the background removed.
        """

        # Remove image background
        return remove(data=image, session=self._session)

    def _find_contours(self, image: np.ndarray, threshold: int):
        """
        Finds contours in the given image using a specified threshold.

        Args:
            image (np.ndarray): The input image.
            threshold (int): The threshold value for contour detection.

        Returns:
            list: A list of contours found in the image.
        """

        # Convert the input image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a binary threshold to create a binary image
        _, thresholded = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image using external retrieval mode and simple approximation
        contours, _ = cv2.findContours(
            thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        return contours

    def get_bounding_box(self, image: np.ndarray) -> List[int]:
        """
        Finds the bounding box coordinates of the main object in the image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            list: A list containing the x, y, width, and height of the bounding box.
        """

        # Remove the background from the input image
        removed = self._remove_background(image)

        # Find contours in the processed image
        contours = self._find_contours(removed, threshold=self._contour_threshold)

        # Identify the contour with the maximum area as the main object
        contour = max(contours, key=cv2.contourArea)

        # Get the bounding rectangle of the main object
        x, y, w, h = cv2.boundingRect(contour)

        return [x, y, w, h]

    def get_bounding_coordinates(self, image: np.ndarray) -> List[List[int]]:
        """
        Finds the edge points of the main object in the image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            list: A list of edge points represented as [x, y] coordinates.
        """

        # Remove the background from the input image
        removed = self._remove_background(image)

        # Find contours in the processed image
        contours = self._find_contours(removed, threshold=self._contour_threshold)

        # Extract edge points from the contours
        edge_points = [
            [x, y] for contour in contours for point in contour for x, y in [point[0]]
        ]

        return edge_points
