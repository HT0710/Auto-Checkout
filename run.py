from omegaconf import DictConfig
import shutil
import hydra

from modules import DatasetGenerator


@hydra.main(config_path="./configs", config_name="run", version_base="1.3")
def main(cfg: DictConfig):
    # Remove the hydra outputs
    shutil.rmtree("outputs")

    # Generate data
    DatasetGenerator(**cfg["data"], **cfg["labeling"]).run()


if __name__ == "__main__":
    main()
