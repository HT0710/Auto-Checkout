from rich import traceback
import yaml
import cv2

from modules.utils import load_config
from modules import Controller

traceback.install()


def check_camera(idx_range: int):
    idx_list = [i for i in range(idx_range) if cv2.VideoCapture(i).isOpened()]

    if len(idx_list) < 3:
        raise ValueError(
            f"The system requires at least 3 cameras to run. Only found {len(idx_list)}."
        )

    count = 0
    cameras = {}

    for idx in idx_list:
        if count == 3:
            break

        cap = cv2.VideoCapture(idx)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            cv2.imshow(str(idx), frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("t"):
                cameras["top_id"] = idx
                count += 1
                break

            elif key == ord("l"):
                cameras["left_id"] = idx
                count += 1
                break

            elif key == ord("r"):
                cameras["right_id"] = idx
                count += 1
                break

        cv2.destroyAllWindows()
        cap.release()

    with open("configs/camera.yaml", "w+") as f:
        yaml.dump(cameras, f)


def main():

    check_camera(10)

    controller = Controller(**load_config("configs/camera.yaml"))

    controller.run()


if __name__ == "__main__":
    main()
