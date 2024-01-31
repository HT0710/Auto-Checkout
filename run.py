from rich import print, traceback
from rich.prompt import Prompt

from modules import DatasetGenerator
from calibration import Calibrate
from utils import load_config

traceback.install()


def main():
    # Calibration
    calibrator = Calibrate(**load_config("configs/calibration.yaml"))

    # Data generate
    data_gen_cfg = load_config("configs/run.yaml")
    data_gen = DatasetGenerator(**data_gen_cfg["data"], **data_gen_cfg["labeling"])

    while True:
        print()
        print("[bold]Auto-Checkout Program[/]")
        print()
        print("1. Dataset generate")
        print("2. Calibration")
        print("3. Quit")
        print()

        choice = Prompt.ask("Enter", choices=["1", "2", "3"])
        print()

        if choice == "1":
            print("Run Dataset Generator")
            data_gen.run()
        elif choice == "2":
            print("Run Calibration")
            print("Preparing")
            calibrator.prepare(camera_id=int(Prompt.ask("Enter Camera ID")))
            print("Calibrating")
            calibrator.run()
        elif choice == "3":
            print("[red]Exit program[/]")
            break
        else:
            continue


if __name__ == "__main__":
    main()
