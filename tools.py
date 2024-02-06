from rich import print, traceback
from rich.prompt import Prompt

from modules import DatasetGenerator, Calibrate
from modules.utils import load_config

traceback.install()


def print_steps(contents: list) -> None:
    for step in contents:
        print(step) if isinstance(step, str) else step()


def main():
    # Calibration
    CALIBRATOR = Calibrate(**load_config("configs/calibration.yaml"))

    # Data generate
    DATAGEN = DatasetGenerator(**load_config("configs/data_gen.yaml"))

    # Menus interface
    MENUS = {
        "main": [
            "-" * 21,
            "[bold]Auto-Checkout Program[/]",
            "-" * 21,
            "1. Calibration",
            "2. Generate dataset",
            "3. Quit\n",
        ],
        "calib": [
            "-" * 15,
            "[bold]> Calibration <[/]\n",
            "1. Preparing...",
            CALIBRATOR.prepare,
            "\n2. Calibrating...",
            CALIBRATOR.run,
            "\n[bold][green]-- Done --[/][/]\n",
        ],
        "data_gen": [
            "-" * 21,
            "[bold]> Dataset Generator <[/]\n",
            DATAGEN.run,
            "\n[bold][green]-- Done --[/][/]\n",
        ],
    }

    # Choices loop
    while True:
        print_steps(MENUS["main"])

        choice = Prompt.ask("Enter", choices=["1", "2", "3"])
        print()

        if choice == "1":
            print_steps(MENUS["calib"])

        elif choice == "2":
            print_steps(MENUS["data_gen"])

        elif choice == "3":
            print("[red]Exit program[/]")
            break

        else:
            continue


if __name__ == "__main__":
    main()
