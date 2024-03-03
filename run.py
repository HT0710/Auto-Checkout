from modules import CameraControler


def main():
    # Define camera controler
    controler = CameraControler(top_ids=[0], side_ids=[2])

    # Run the controler
    controler.run()


if __name__ == "__main__":
    main()
