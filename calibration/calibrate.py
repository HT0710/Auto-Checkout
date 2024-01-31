from typing import Tuple, Union
from pathlib import Path
import shutil

from rich.prompt import Prompt
from rich import print
import numpy as np
import cv2

from utils import tuple_handler


class Calibrate:
    def __init__(
        self, grid_size: Tuple[int], frame_size: Tuple[int], diameter: int, refine: bool
    ) -> None:
        """
        Initializes a calibration object with specified parameters.

        Args:
            grid_size (Tuple[int]): The size of the grid in (rows, columns).
            frame_size (Tuple[int]): The size of the frame or image in (height, width).
            diameter (int): The diameter of the circles in the calibration pattern.
            refine (bool): A flag indicating whether to refine corner positions.
        """
        self.grid_size = tuple_handler(grid_size, max_dim=2)
        self.frame_size = tuple_handler(frame_size, max_dim=2)
        self.diameter = int(diameter)
        self.refine = bool(refine)
        self.images_path = Path("calibration/images")
        self.save_path = Path("calibration/calibration_params.npz")

    def _detect_board(self, image: np.ndarray) -> Union[np.ndarray, None]:
        """
        Detects the calibration pattern in the input image.

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            np.ndarray: Corner positions if the pattern is detected, otherwise None.
        """

        # Convert to Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect pattern
        ret, corners = cv2.findCirclesGrid(
            gray, self.grid_size, None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID
        )

        # Check detect status
        if not ret:
            return

        # Termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Refine corners
        if self.refine:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

        return corners

    def prepare(self, camera_id: int) -> None:
        """
        Prepares the calibration by capturing images from the specified camera.

        Args:
            camera_id (int): ID of the camera to be used for calibration.

        Raises:
            ValueError: If the camera cannot be opened.
        """

        # Capture camera
        cap = cv2.VideoCapture(camera_id)

        # Check capture
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera id: {camera_id}")

        # Check images path
        if self.images_path.exists() and any(self.images_path.iterdir()):
            confirm = Prompt.ask("Delete old calibrate images? (y/N)").strip().lower()

            if confirm in ["y", "yes"]:
                shutil.rmtree(str(self.images_path))
            else:
                return

        # Prepare
        i = 0
        self.images_path.mkdir(parents=True, exist_ok=True)

        # Open camera
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Detect board
            corners = self._detect_board(image=frame)

            # Check detect status
            if corners is not None:
                # Draw and display the corners
                review = cv2.drawChessboardCorners(
                    frame.copy(), self.grid_size, corners, True
                )
            else:
                review = frame

            # Add help infomation to frame
            for text, pos, color in [
                (
                    f"Image count: {len(list(self.images_path.glob('*.jpg')))}",
                    (10, 50),
                    (255, 0, 255),
                ),
                ("s: Save image", (10, 150), (255, 0, 0)),
                ("q: Quit", (10, 200), (0, 0, 255)),
            ]:
                cv2.putText(review, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Review
            cv2.imshow(f"Camera {camera_id}", review)

            # Delay
            key = cv2.waitKey(1) & 0xFF

            # Check Quit signal
            if key == ord("q"):
                break

            # Check Save signal
            elif key == ord("s"):
                # Check detection
                if corners is None:
                    print("Cannot detect pattern, image is not saved.")
                    continue

                # Save image
                cv2.imwrite(f"{self.images_path}/{i}.jpg", frame)
                i += 1

    def run(self) -> None:
        """
        Runs the camera calibration process using captured images.

        Raises:
            RuntimeError: If the calibration process fails unexpectedly.
        """

        # Initialize 3D array of zeros
        pattern_points = np.zeros(
            (self.grid_size[0] * self.grid_size[1], 3), np.float32
        )

        # Fill the calibration pattern's grid
        pattern_points[:, :2] = np.mgrid[
            0 : self.grid_size[0], 0 : self.grid_size[1]
        ].T.reshape(-1, 2)

        # Convert the coordinates from pixel units to mm
        pattern_points = pattern_points * self.diameter

        # Arrays to store object points and image points
        object_points = []
        image_points = []

        # Get all images
        images = self.images_path.glob("*.jpg")

        if not list(images):
            raise ValueError("Cannot found any images for calibration.")

        for image in images:
            image = cv2.imread(str(image))

            # Detect pattern
            corners = self._detect_board(image)

            if corners is None:
                continue

            # Update params
            object_points.append(pattern_points)
            image_points.append(corners)

        # Calibrating
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, self.frame_size, None, None
        )

        if ret:
            # Save calibration parameters
            np.savez(
                file=str(self.save_path), mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs
            )
        else:
            raise RuntimeError("Unexpected error. Calibrate failed!")
