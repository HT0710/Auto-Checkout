from modules import TopEngine, SideEngine
from modules.utils import load_config


def main():
    top_camera = TopEngine(
        camera_ids=[0], engine_configs=load_config("configs/engine/top.yaml")
    )
    # side_camera = SideEngine(
    #     camera_ids=[1, 2], engine_configs=load_config("configs/engine/side.yaml")
    # )

    top_camera.run()
    # side_camera.run()


if __name__ == "__main__":
    main()
