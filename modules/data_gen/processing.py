from typing import List, Tuple, Union
from types import GeneratorType
import itertools
import os

import numpy as np
import cv2

from ..utils import tuple_handler


class ImageProcessing:
    @staticmethod
    def resize(
        image: np.ndarray, size: Union[int, List[int], Tuple[int]]
    ) -> np.ndarray:
        """
        Resize image to the specified dimensions.

        Args:
            image (np.ndarray): Input image as a NumPy array.
            size (Union[int, List[int], Tuple[int]]): A tuple specifying the target size (width, height).

        Returns:
            np.ndarray: A NumPy array representing the resized image.
        """
        return cv2.resize(image, tuple_handler(size, 2), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def center_crop(image: np.ndarray) -> np.ndarray:
        """
        Performs a center crop on the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image as a NumPy array.

        Returns
        -------
        np.ndarray
            The center-cropped image.
        """
        img_h, img_w = image.shape[:2]

        top = (img_h - img_w) // 2 if img_h > img_w else 0

        bottom = top + img_w if img_h > img_w else img_h

        left = (img_w - img_h) // 2 if img_w > img_h else 0

        right = left + img_h if img_w > img_h else img_w

        return image[top:bottom, left:right]

    @staticmethod
    def crop_border(image: np.ndarray, crop_rate: int) -> np.ndarray:
        """
        Removes borders from the input image based on the specified crop rate.

        Parameters
        ----------
        image : np.ndarray
            The input image as a NumPy array.
        crop_rate : int
            The percentage of border to be cropped from each side of the image.
            For example, if `crop_rate` is 10, it will remove 10% of the border
            from each side of the image.

        Returns
        -------
        np.ndarray
            The image with borders cropped based on the specified crop rate.
        """
        img_h, img_w = image.shape[:2]

        crop_h, crop_w = map(int, (x * crop_rate / 100 for x in (img_h, img_w)))

        return image[crop_h : img_h - crop_h, crop_w : img_w - crop_w]

    @staticmethod
    def add_border(
        image: np.ndarray, border_color: Union[Tuple, int] = (0, 0, 0)
    ) -> np.ndarray:
        """
        Adds a border around a given video frame to make it a square frame.

        Args:
            video (np.ndarray): Input video frame as a NumPy array.
            border_color (Tuple|int): Color of the border. Defaults to (0, 0, 0) for black.

        Returns:
            np.ndarray: A NumPy array representing the video after add border.
        """
        img_h, img_w = image.shape[:2]
        target_size = max(img_h, img_w)

        border_v = (target_size - img_h) // 2
        border_h = (target_size - img_w) // 2

        return cv2.copyMakeBorder(
            image,
            border_v,
            border_v,
            border_h,
            border_h,
            cv2.BORDER_CONSTANT,
            border_color,
        )


class VideoProcessing:
    """
    Process included:

    Load: Load video and return list of frames.
    Sampling: Choose 1 frame every n frames.
    Resize: Change size of a video.
    """

    @staticmethod
    def load(path: str, rgb: bool = False) -> List[np.ndarray]:
        """
        Load a video file and return it as a NumPy array.

        Args:
            path (str): The file path to the video.
            rgb (bool): Convert the video to RGB format.

        Returns:
            List[np.ndarray]: A NumPy array representing the video frames.

        Raises:
            - FileExistsError: If the specified file does not exist.
            - RuntimeError: If the video file cannot be opened.
        """
        # Check path
        if not os.path.exists(path):
            raise FileExistsError("File not found!")
        # Load video
        video = cv2.VideoCapture(path)
        # Check video
        if not video.isOpened():
            raise RuntimeError("Could not open video file.")
        # Extract frames
        output = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if rgb else frame
            for _, frame in iter(video.read, (False, None))
        )
        # video.release()
        return output

    @staticmethod
    def subsample(
        video: Union[np.ndarray, List[np.ndarray], GeneratorType], value: int
    ) -> List[np.ndarray]:
        """
        Perform frame subsampling on a video.

        Args:
            video (Union[np.ndarray, List[np.ndarray]]): Input video as a NumPy array.
            value (int): The sampling value. If 0, no sampling is performed.

        Returns:
            List[np.ndarray]: A NumPy array representing the sampled video frames.
        """
        if isinstance(video, GeneratorType):
            output = itertools.islice(video, 0, None, value)
        elif isinstance(video, list):
            output = video[::value]
        return output

    @staticmethod
    def resize(
        video: Union[np.ndarray, List[np.ndarray], GeneratorType],
        size: Union[int, List[int], Tuple[int]],
    ) -> List[np.ndarray]:
        """
        Resize each frame of video to the specified dimensions.

        Args:
            video (Union[np.ndarray, List[np.ndarray]]): Input video as a NumPy array.
            size (Union[int, List[int], Tuple[int]]): A tuple specifying the target size (width, height).

        Returns:
            List[np.ndarray]: A NumPy array representing the resized video.
        """
        return (cv2.resize(frame, tuple_handler(size, 2)) for frame in video)
