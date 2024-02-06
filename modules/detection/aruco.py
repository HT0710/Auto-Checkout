from typing import Tuple

import cv2.aruco as aruco
import numpy as np
import cv2


class ArucoDetector:
    def __init__(self, dictionary_type: str, size: int = 1000) -> None:
        """
        Initializes an ArucoDetector object.

        Parameters
        ----------
        dictionary_type : str
            The type of ArUco dictionary to be used.
        size : int, optional
            The size of the ArUco dictionary, by default 1000.
        """
        self._detector = aruco.ArucoDetector(
            dictionary=aruco.getPredefinedDictionary(
                getattr(aruco, f"DICT_{dictionary_type.strip().upper()}_{size}")
            )
        )

    def detect(self, image: np.ndarray) -> Tuple[Tuple[np.ndarray], np.ndarray]:
        """
        Detects ArUco markers in the given image.

        Parameters
        ----------
        image : numpy.ndarray
            The input image containing ArUco markers.

        Returns
        -------
        Tuple[Tuple[np.ndarray], np.ndarray]
            A tuple containing the corners of detected markers and their corresponding ids.
        """

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = self._detector.detectMarkers(gray)

        return corners, ids

    def draw(
        self, image: np.ndarray, corners: Tuple[np.ndarray], ids: np.ndarray
    ) -> np.ndarray:
        """
        Draws ArUco markers on the given image.

        Parameters
        ----------
        image : numpy.ndarray
            The input image on which to draw the ArUco markers.
        corners : Tuple[np.ndarray]
            The corners of detected ArUco markers.
        ids : np.ndarray
            The ids of detected ArUco markers.

        Returns
        -------
        numpy.ndarray
            The image with ArUco markers drawn on it.
        """
        return aruco.drawDetectedMarkers(image, corners, ids)
