from functools import cached_property
from pathlib import Path, PosixPath
from typing import Any, List, Tuple
from datetime import datetime
import random
import shutil
import psutil

from tqdm import tqdm
import cv2

from ..utils import device_handler, tuple_handler
from .processing import ImageProcessing
from .labeling import AutoLabeling


class DatasetGenerator:
    def __init__(
        self,
        data_path: str = "video",
        save_path: str = "dataset",
        save_name: str = None,
        image_size: int = 640,
        split_size: Tuple = (0.7, 0.2, 0.1),
        model: str = "silueta",
        device: str = "auto",
        tensorrt: bool = False,
    ) -> None:
        """
        Initialize DatasetGenerator.

        Args:
            data_path (str): Path to the video data.
            save_path (str): Path to save the generated dataset.
            image_size (int): Size to which the frames should be resized.
            split_size (Tuple): Tuple representing train-validation-test split ratios.
            model (str): Model for labeling configuration.
            device (str): Device for processing (e.g., "auto", "cpu", "cuda").
            tensorrt (bool): Flag indicating whether to use TensorRT for inference.
        """
        self.data_path = Path(data_path)
        self.save_path = self._check_save(save_path, save_name)
        self.image_size = image_size
        self.split_size = tuple_handler(split_size, max_dim=3)
        self.labeler = AutoLabeling(
            model=model, device=device_handler(device), tensorrt=tensorrt
        )
        self.image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

    @cached_property
    def classes(self):
        """
        Returns a sorted list of class names based on the folders present in the data path.

        Returns
        -------
        list
            A sorted list of class names.
        """
        return sorted(
            [folder.name for folder in self.data_path.iterdir() if folder.is_dir()]
        )

    def _check_save(self, path: str, name: str) -> PosixPath:
        """
        Check and format the save path.

        Args:
            path (str): The directory path where the file will be saved.
            name (str): The name of the file.

        Returns:
            PosixPath: The formatted save path.
        """

        # Check name
        if not name:
            name = datetime.now().strftime("%d-%b-%y")

        return Path(path) / name

    def _list_image(self, path: Path):
        """
        Recursively lists image files with extensions specified in self.image_extensions.

        Parameters
        ----------
        path : Path
            The path to search for image files.

        Returns
        -------
        list
            A list of Path objects representing image files.
        """
        return [
            image for ext in self.image_extensions for image in path.rglob("*" + ext)
        ]

    def _process_images(self) -> None:
        """Process the image by resizing it and saving it to the specified save path"""

        # List all image path
        data = self._list_image(path=self.data_path)

        # Process loop
        for path in tqdm(data, desc="Process image", colour="cyan"):
            # Load
            image = cv2.imread(str(path))

            # Reize
            image = ImageProcessing.resize(image, self.image_size)

            # Create save folder
            self.save_path.mkdir(parents=True, exist_ok=True)

            # Setup save name
            save_name = f"{path.parent.name}@{path.name}"

            # Save image
            cv2.imwrite(filename=str(self.save_path / save_name), img=image)

    def _split_data(self) -> None:
        """Split the data into training, validation, and test sets and move them to corresponding folders"""

        # Create save folders
        save_folders = [self.save_path / folder for folder in ["train", "val", "test"]]
        [folder.mkdir(exist_ok=True) for folder in save_folders]

        # Get image list
        data = self._list_image(path=self.save_path)

        # Shuffle data
        random.shuffle(data)

        # Split data
        split_sizes = [round(len(data) * size) for size in self.split_size]
        splited_paths = [
            data[: split_sizes[0]],
            data[split_sizes[0] : split_sizes[0] + split_sizes[1]],
            data[split_sizes[0] + split_sizes[1] :],
        ]

        # Define progress bar
        progress = tqdm(total=len(data), desc="Split data", colour="cyan")

        # Move to new destination
        for data_paths, save_folder in zip(splited_paths, save_folders):
            for path in data_paths:
                shutil.move(src=str(path), dst=str(save_folder / path.name))
                progress.update()

    def _generate_label(self) -> None:
        """Generate labels for the images and move them to corresponding folders"""

        # List all image path
        data = self._list_image(path=self.save_path)

        # Iter data
        for path in tqdm(data, desc="Generate label", colour="cyan"):
            # Define save path
            images_folder = path.parent / "images"
            labels_folder = path.parent / "labels"

            # Create save path
            images_folder.mkdir(exist_ok=True)
            labels_folder.mkdir(exist_ok=True)

            # Read image
            image = cv2.imread(str(path))

            # Get label
            x, y, w, h = self.labeler.get_bounding_box(image)

            # Calculate center
            x_center, y_center = (x + w / 2), (y + h / 2)

            # Create label info
            info = [str(i / self.image_size) for i in [x_center, y_center, w, h]]

            # Move image
            shutil.move(src=str(path), dst=str(images_folder / path.name))

            # Create label file
            with open(str(labels_folder / path.stem) + ".txt", "w+") as f:
                class_name = path.stem.split("@")[0]
                f.write(f"{self.classes.index(class_name)} {' '.join(info)}")

    def run(self):
        """
        Execute the data processing pipeline:

            1. Image processing
            2. Data splitting
            3. Label generation
            4. Create data.yaml
        """

        # ----------------
        # 1. Process image
        self._process_images()

        # -------------
        # 2. Split data
        self._split_data()

        # -----------------
        # 3. Generate label
        self._generate_label()

        # -------------------
        # 4. Create data.yaml
        with open(str(self.save_path / "data.yaml"), "w+") as f:
            f.write(
                "\n".join(
                    [
                        "# Configuration for dataset\n",
                        f"path: {self.save_path.resolve()}\n",
                        "train: ../train/images",
                        "val: ../val/images",
                        "test: ../test/images\n",
                        f"nc: {len(self.classes)}",
                        f"names: {self.classes}",
                    ]
                )
            )
