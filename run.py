from pathlib import Path

import cv2

from processing import VideoProcessing
from labeling import AutoLabeling


def main():
    # Define modules
    PROCESSER = VideoProcessing()
    # LABELER = AutoLabeling(model="u2net")

    # Data path
    DATA_PATH = Path("video")

    SAVE_PATH = Path("data")

    # Get all folders
    folders = (folder for folder in DATA_PATH.iterdir() if folder.is_dir())

    # Iter folders
    for folder in folders:
        # Get all videos
        files = (file for file in folder.iterdir() if file.is_file())

        # Iter videos
        for file in files:
            # Process video
            # Load
            video = PROCESSER.load(str(file))

            # Subsampling
            video = PROCESSER.subsample(video, 3)

            # Resize
            video = PROCESSER.resize(video, 640)

            for frame in video:
                # x, y, w, h = LABELER.get_bounding_box(frame)

                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                cv2.imshow("abc", frame)

                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break


if __name__ == "__main__":
    main()
