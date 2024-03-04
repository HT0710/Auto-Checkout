from rich import traceback

from modules import CameraControler

traceback.install()


def main():
    # Define camera controler
    controler = CameraControler(top_ids=[], side_ids=[0])

    # Run the controler
    controler.run()


if __name__ == "__main__":
    main()
