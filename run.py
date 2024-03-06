from rich import traceback
import yaml
import cv2

from modules.utils import load_config
from modules import CameraControler

traceback.install()


def check_camera():
    idx_list = [i for i in range(10) if cv2.VideoCapture(i).isOpened()]

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

    check_camera()

    # Define camera controler
    controller = CameraControler(**load_config("configs/camera.yaml"))

    # Run the controler
    controller.run()


if __name__ == "__main__":
    main()
