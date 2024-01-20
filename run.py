from pathlib import Path

import cv2

from modules import VideoProcessing, AutoLabeling


def main():
    # -------------
    # Configuration
    PROCESSER = VideoProcessing()

    LABELER = AutoLabeling(model="silueta", device="auto", tensorrt=False)

    DATA_PATH = Path("video")

    SAVE_PATH = Path("data")

    IMAGE_SIZE = 640

    # -----
    # Setup

    # Define data folder
    images_path = SAVE_PATH / "images"
    labels_path = SAVE_PATH / "labels"

    # Create data folder
    images_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)

    # Check classes
    class_folders = sorted(
        [folder for folder in DATA_PATH.iterdir() if folder.is_dir()]
    )
    classes = {folder.name: i for i, folder in enumerate(class_folders)}

    # Check all videos
    extensions = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".mpg")
    video_paths = (video for ext in extensions for video in DATA_PATH.rglob("*" + ext))

    # Iter videos
    for video_path in video_paths:
        # Process video
        # Load
        video = PROCESSER.load(path=str(video_path))

        # Subsampling
        video = PROCESSER.subsample(video=video, value=3)

        # Resize
        video = PROCESSER.resize(video=video, size=IMAGE_SIZE)

        # Iter frames
        for i, frame in enumerate(video):
            # Create save name
            formatted_name = f"{video_path.parent.name}_{video_path.stem}_{i}"

            # Create save path
            image_path = f"{images_path}/{formatted_name}.jpg"
            label_path = f"{labels_path}/{formatted_name}.txt"

            # Save image
            cv2.imwrite(filename=image_path, img=frame)

            # Get normalized label
            label = [str(i / IMAGE_SIZE) for i in LABELER.get_bounding_box(frame)]

            # Save label
            with open(label_path, "w+") as f:
                f.write(f"{classes[video_path.parent.name]} {' '.join(label)}")


if __name__ == "__main__":
    main()
