from rich import traceback

from modules.utils import load_config
from modules import CameraControler

traceback.install()


def main():
    # Define camera controler
    controller = CameraControler(**load_config("configs/camera.yaml"))

    # Run the controler
    controller.run()


if __name__ == "__main__":
    main()
