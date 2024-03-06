from functools import cache, cached_property
from typing import Callable, Dict, Tuple
from collections import deque
from datetime import datetime
from pathlib import Path
import math
import time

from rich import print
import numpy as np
import cv2

from ..utils import tuple_handler


class Camera:
    def __init__(
        self,
        device_id: int,
        delay: int = 1,
        subsampling: int = 1,
        resolution: Tuple[int] = None,
        calibration: str = None,
        record: Dict = None,
        debug: Dict = None,
    ) -> None:
        """
        Initializes a camera object.

        Parameters
        ----------
        device_id : int
            Identifier for the camera device.
        delay : int, optional
            Time delay (in seconds) between capturing frames, by default 1.
        subsampling : int, optional
            Rate at which frames are subsampled, by default 1.
        resolution : Tuple[int], optional
            Resolution of the captured frames (width, height), by default None.
        calibration : str, optional
            Path to camera parameters for calibration, by default None.
        record : Dict, optional
            Configuration for recording functionality, by default None.
        debug : Dict, optional
            Configuration for debugging functionality, by default None.
        """
        self.device_id = device_id
        self.wait = delay
        self.capture = self._check_capture(device_id)
        self.subsampling = max(1, int(subsampling))
        self.resolution = tuple_handler(resolution, max_dim=2) if resolution else None
        self._setup_calibration(path=calibration)
        if record:
            self._setup_recorder(**record)
        self._setup_debuger(**debug)

    def __len__(self) -> int:
        """
        Returns the total number of frames in the video capture.

        Returns
        -------
        int
            Total number of frames in the video capture.
        """
        return int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> "Camera":
        """Iterates over the frames captured by the camera."""

        # Video iteration generate
        def generate():
            for _, frame in iter(self.capture.read, (False, None)):
                yield frame

        # Initialize frames queue
        self.queue = generate()

        # Initialize progress
        self.current_progress = 0

        return self

    def __tracker(func):
        """Decorator for tracking the performance of functions."""

        # Create wrapper function
        def wrapper(self):
            # Check on first run
            if not hasattr(self, "start_time"):
                self.start_time = time.time()

            # Run the function
            output = func(self)

            # Get delay time
            delay = time.time() - self.start_time

            # Check fps debuger
            if hasattr(self, "fps_history"):
                self.fps_history.append(math.ceil(1 / delay))
                print(f"FPS: {round(np.mean(self.fps_history))}", end="\r")

            # Setup for new circle
            self.start_time = time.time()

            # Return function output
            return output

        return wrapper

    @__tracker
    def __next__(self) -> np.ndarray:
        """
        Returns the next frame from the camera capture.

        Returns
        -------
        numpy.ndarray
            The next frame in the camera capture.
        """

        # Get current frame
        self.current_frame = next(self.queue)

        # Camera calibration
        if hasattr(self, "camera_params"):
            self.current_frame = self._undistort(self.current_frame)

        # Change video resolution
        if self.resolution:
            self.current_frame = cv2.resize(self.current_frame, self.resolution)

        # Recorder the video
        if hasattr(self, "recorder"):
            if hasattr(self, "record_res"):
                save_frame = cv2.resize(self.current_frame, self.record_res)
            else:
                save_frame = self.current_frame
            self.recorder.write(save_frame)

        # Update progress
        self.current_progress += 1

        # Return current frame
        return self.current_frame

    def _check_capture(self, value: int) -> cv2.VideoCapture:
        """
        Checks if the camera capture is successfully opened.

        Parameters
        ----------
        value : int
            The camera device ID.

        Returns
        -------
        cv2.VideoCapture
            The VideoCapture object representing the camera capture.

        Raises
        ------
        ValueError
            If the camera capture cannot be opened.
        """

        # Capture video
        cap = cv2.VideoCapture(value)

        # Check capture
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera ID: {value}")

        return cap

    def _setup_calibration(self, path: str) -> None:
        """
        Sets up calibration parameters from a given file path.

        Parameters
        ----------
        path : str
            The file path to the calibration data in .npz format.
        """

        # Check path
        if not path or (not Path(path).is_file() and not path.endswith(".npz")):
            return

        # Load camera parameters
        self.camera_params = np.load(path)

    def _setup_recorder(self, path: str, resolution: Tuple[int]) -> None:
        """
        Sets up the video recorder for saving the captured frames.

        Parameters
        ----------
        path : str
            The directory path where the recorded video will be saved.
        resolution : Tuple[int]
            The resolution of the recorded video (width, height).
        """

        # Set save folder
        save_folder = Path(path)
        # Create save folder
        save_folder.mkdir(parents=True, exist_ok=True)
        # Create save name
        save_name = datetime.now().strftime("%d-%b-%y") + ".mp4"
        # Create save path
        save_path = str(save_folder / save_name)
        # Setup codec
        codec = cv2.VideoWriter_fourcc(*"mp4v")

        # Setup record resolution
        self.record_res = resolution or self.size()

        # Config writer
        self.recorder = cv2.VideoWriter(
            filename=save_path, fourcc=codec, fps=self.fps, frameSize=self.record_res
        )

    def _setup_debuger(self, fps: bool) -> None:
        """
        Sets up the debugger for monitoring camera performance.

        Parameters
        ----------
        fps : bool
            Flag indicating whether to monitor frames per second (fps).
        """

        # Check fps
        if fps:
            self.fps_history = deque(maxlen=30)

    def _undistort(self, frame: np.ndarray) -> np.ndarray:
        """
        Undistorts a given image frame using camera calibration parameters if available.

        Parameters
        ----------
        frame : np.ndarray
            The input image frame to undistort.

        Returns
        -------
        np.ndarray
            The undistorted image frame.
        """

        # Get camera params
        mtx = self.camera_params["mtx"]
        dist = self.camera_params["dist"]

        # Get shape
        h, w = frame.shape[:2]

        # Get optimal matrix
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1)

        # Undistort image
        map_x, map_y = cv2.initUndistortRectifyMap(
            mtx, dist, None, new_mtx, (w, h), cv2.CV_32FC1
        )

        # Remap image
        frame = cv2.remap(frame, map_x, map_y, cv2.INTER_CUBIC)

        x, y, w, h = roi

        return frame[y : y + h, x : x + w]

    @cached_property
    def name(self) -> str:
        """
        Returns the name of the camera device.

        Returns
        -------
        str
            The name of the camera device.
        """
        return str(self.device_id)

    @cached_property
    def fps(self) -> int:
        """
        Returns the frames per second (fps) of the camera capture.

        Returns
        -------
        int
            The frames per second (fps) of the camera capture.
        """
        return int(self.capture.get(cv2.CAP_PROP_FPS))

    @cache
    def size(self, reverse: bool = False) -> Tuple[int]:
        """
        Returns the size (width, height) of the camera capture.

        Parameters
        ----------
        reverse : bool, optional
            Flag indicating whether to return the size in (height, width) format,
            by default False.

        Returns
        -------
        Tuple[int]
            The size of the camera capture.
        """
        if self.resolution:
            w, h = self.resolution
        else:
            w, h = (
                int(self.capture.get(getattr(cv2, f"CAP_PROP_FRAME_{prop}")))
                for prop in ["WIDTH", "HEIGHT"]
            )

        return (w, h) if not reverse else (h, w)

    def delay(self, value: int = None) -> bool:
        """
        Delays the execution for a specified time and checks for a key press.

        Parameters
        ----------
        value : int, optional
            The delay time in milliseconds, by default None.

        Returns
        -------
        bool
            True if no key 'q' is pressed within the delay time, otherwise False.
        """
        key = cv2.waitKey(max(0, value if value else self.wait)) & 0xFF

        # Check continue
        return True if not key == ord("q") else False

    def release(self) -> None:
        """Release capture"""

        # Main capture
        self.capture.release()

        # Recorder
        if hasattr(self, "recorder"):
            self.recorder.release()

        # Finish up
        cv2.destroyWindow(self.name)

    def run(self, callback: Callable = None) -> None:
        """
        Runs the camera capture loop, displaying frames and optionally applying a callback function.

        Parameters
        ----------
        callback : Callable, optional
            A function to apply to each frame, by default None.
        """

        # Playback
        for frame in self:

            # Check callback
            frame = callback(frame) if callback else self.current_frame

            # Show frame
            cv2.imshow(self.name, frame)

            # Delay
            if not self.delay(self.wait):
                break

        # Release
        self.release()
