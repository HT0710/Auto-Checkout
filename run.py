from modules.utils import load_config
from modules import Camera


def main():
    top = Camera(**load_config("configs/camera.yaml"))

    top.run()


if __name__ == "__main__":
    main()
