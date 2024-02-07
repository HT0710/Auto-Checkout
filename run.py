from modules import TopEngine
from modules.utils import load_config


def main():
    top_camera = TopEngine(
        camera_ids=[0], camera_configs=load_config("configs/camera.yaml")
    )

    top_camera.run()


if __name__ == "__main__":
    main()
