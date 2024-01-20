from pathlib import Path, PosixPath
from typing import List, Tuple
import random
import shutil
import psutil

from tqdm.contrib.concurrent import process_map
from rembg import remove, new_session
from tqdm import tqdm
import numpy as np
import cv2

from .processing import VideoProcessing
from .utils import *


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


class DatasetGenerator:
    def __init__(
        self,
        data_path: str = "video",
        save_path: str = "dataset",
        subsample: int = 3,
        image_size: int = 640,
        split_size: Tuple = (0.7, 0.2, 0.1),
        num_workers: int = 0,
        model: str = "silueta",
        device: str = "auto",
        tensorrt: bool = False,
    ) -> None:
        """
        Initialize DatasetGenerator.

        Args:
            data_path (str): Path to the video data.
            save_path (str): Path to save the generated dataset.
            subsample (int): Subsampling factor for video frames.
            image_size (int): Size to which the frames should be resized.
            split_size (Tuple): Tuple representing train-validation-test split ratios.
            num_workers (int): Number of workers for parallel processing.
            model (str): Model for labeling configuration.
            device (str): Device for processing (e.g., "auto", "cpu", "cuda").
            tensorrt (bool): Flag indicating whether to use TensorRT for inference.
        """
        self.data_path = Path(data_path)
        self.save_path = Path(save_path)
        self.subsample = subsample
        self.image_size = image_size
        self.split_size = tuple_handler(split_size, max_dim=3)
        self.workers = workers_handler(num_workers)
        self.labeling_config = {"model": model, "device": device, "tensorrt": tensorrt}
        self.video_extensions = [".mp4", ".avi", ".mkv", ".mov", ".flv", ".mpg"]
        self.image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

    def _process_video(self, path: PosixPath) -> None:
        """
        Process the video and save frames.

        Args:
            path (PosixPath): Path to video to be processed.
        """

        # Load
        video = VideoProcessing.load(str(path))
        # Subsample
        video = VideoProcessing.subsample(video, value=self.subsample)
        # Resize
        video = VideoProcessing.resize(video, size=self.image_size)

        # Create save path
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Save frames
        for i, frame in enumerate(video):
            save_name = f"{path.parent.name}_{path.stem}_{i}.jpg"

            cv2.imwrite(filename=f"{self.save_path}/{save_name}", img=frame)

    def _split_data(self, data: List[PosixPath]) -> None:
        """
        Split the data into training, validation, and test sets and move them to corresponding folders.

        Args:
            data (List[PosixPath]): List of PosixPath objects representing the paths of data to be split.
        """

        # Create save folders
        save_folders = [self.save_path / folder for folder in ["train", "val", "test"]]
        [folder.mkdir(exist_ok=True) for folder in save_folders]

        # Shuffle data
        random.shuffle(data)

        # Split into train, val, test chunk
        split_sizes = [round(len(data) * size) for size in self.split_size]
        splited_paths = [
            data[: split_sizes[0]],
            data[split_sizes[0] : split_sizes[0] + split_sizes[1]],
            data[split_sizes[0] + split_sizes[1] :],
        ]

        # Define progress bar
        progress = tqdm(total=len(data), desc=f"Split data", colour="cyan")

        # Move to new destination
        for data_paths, save_folder in zip(splited_paths, save_folders):
            for path in data_paths:
                shutil.move(src=str(path), dst=str(save_folder / path.name))
                progress.update()

    def _generate_label(self, data: List[PosixPath]) -> None:
        """
        Generate labels for the images and move them to corresponding folders.

        Args:
            data (List[PosixPath]): List of PosixPath objects representing the paths of images to generate labels.
        """

        # Iter data
        for path in tqdm(data, desc=f"Generate label", colour="cyan"):
            # Define save path
            images_folder = path.parent / "images"
            labels_folder = path.parent / "labels"

            # Create save path
            images_folder.mkdir(exist_ok=True)
            labels_folder.mkdir(exist_ok=True)

            # Read image
            image = cv2.imread(str(path))

            # Get label
            label = [
                str(i / self.image_size) for i in self.labeler.get_bounding_box(image)
            ]

            # Move image
            shutil.move(src=str(path), dst=str(images_folder / path.name))

            # Create label file
            with open(str(labels_folder / path.stem) + ".txt", "w+") as f:
                class_name = path.stem.split("_")[0]
                f.write(f"{self.classes.index(class_name)} {' '.join(label)}")

    def run(self):
        """
        Execute the data processing pipeline:

            1. Video processing
            2. Data splitting
            3. Label generation
            4. Create data.yaml
        """
        # Calcute chunksize based on cpu parallel power
        benchmark = lambda x: max(
            1, round(len(x) / (self.workers * psutil.cpu_freq().max / 1000) / 4)
        )

        # ----------------
        # 1. Process video

        # Get all videos path
        video_paths = [
            video
            for ext in self.video_extensions
            for video in self.data_path.rglob("*" + ext)
        ]

        # Process all videos
        process_map(
            self._process_video,
            video_paths,
            max_workers=self.workers,
            chunksize=benchmark(video_paths),
            desc="Process video",
            colour="cyan",
        )

        # -------------
        # 2. Split data

        # Get all images path
        images_path = [
            image
            for ext in self.image_extensions
            for image in self.save_path.rglob("*" + ext)
        ]

        # Split data
        self._split_data(data=images_path)

        # -----------------
        # 3. Generate frame

        # Get all images path (again)
        images_path = [
            image
            for ext in self.image_extensions
            for image in self.save_path.rglob("*" + ext)
        ]

        # Get all classes
        self.classes = sorted(
            [folder.name for folder in self.data_path.iterdir() if folder.is_dir()]
        )

        # Define labeler
        self.labeler = AutoLabeling(**self.labeling_config)

        # Generate label
        self._generate_label(data=images_path)

        # -------------------
        # 4. Create data.yaml

        with open(str(self.save_path / "data.yaml"), "w+") as f:
            f.write(
                "\n".join(
                    [
                        f"path: {self.save_path.resolve()}",
                        f"train: ../train/images",
                        f"val: ../val/images",
                        f"test: ../test/images",
                        f"nc: {len(self.classes)}",
                        f"names: {self.classes}",
                    ]
                )
            )
