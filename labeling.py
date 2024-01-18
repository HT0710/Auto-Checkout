from rembg import remove, new_session
import numpy as np
import cv2


class AutoLabeling:
    def __init__(self, model: str = "silueta") -> None:
        """
        Initializes an instance of the AutoLabeling class.

        Args:
            model (str, optional): The name of the model to be used. Defaults to "silueta".

        For more models information: https://github.com/danielgatis/rembg
        """
        self.session = new_session(model)

    def remove_background(self, image: np.ndarray):
        """
        Removes the background from the input image using the initialized model.

        Args:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The image with the background removed.
        """
        return remove(data=image, session=self.session)

    def find_contours(self, image: np.ndarray, threshold: int):
        """
        Finds contours in the given image using a specified threshold.

        Args:
            image (np.ndarray): The input image.
            threshold (int): The threshold value for contour detection.

        Returns:
            list: A list of contours found in the image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def get_bounding_box(self, image):
        """
        Finds the bounding box coordinates of the main object in the image.

        Args:
            image: The input image.

        Returns:
            list: A list containing the x, y, width, and height of the bounding box.
        """
        removed = self.remove_background(image)
        contours = self.find_contours(removed, threshold=20)
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        return [x, y, w, h]

    def get_bounding_coordinates(self, image):
        """
        Finds the edge points of the main object in the image.

        Args:
            image: The input image.

        Returns:
            list: A list of edge points represented as (x, y) coordinates.
        """
        removed = self.remove_background(image)
        contours = self.find_contours(removed, threshold=20)
        edge_points = [
            (x, y) for contour in contours for point in contour for x, y in [point[0]]
        ]
        return edge_points
