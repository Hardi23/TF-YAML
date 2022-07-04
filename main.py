# noinspection UnusedImport
import sys
import warnings

import environ_config
import hashlib
import logging
import os
from pathlib import Path

if not environ_config.check_requirements():
    warnings.warn(f"Some required packages seem to be missing, please install!")
    print("Exiting")
    sys.exit(-1)

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from inputtr import inputter

from training_runner import TrainingRunner
from config import FullConfig, CONFIG_PATH, CONFIG_NAME

inputter.nt_disable_colors = False
cs = ConfigStore.instance()
cs.store(name="FullConfig", node=FullConfig)


def check_config_file_exists():
    config_file = Path(os.path.join(os.getcwd(), CONFIG_PATH, CONFIG_NAME))
    if not config_file.exists():
        config_file.parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(os.path.dirname(config_file)):
            config_file.touch(exist_ok=True)
    return config_file.exists()


def check_required_configurations(cfg: FullConfig):
    if not cfg.paths.training or not os.path.exists(cfg.paths.training):
        inputter.print_error("Path to training pictures is missing!")
        return False
    elif len(os.listdir(cfg.paths.training)) == 0:
        inputter.print_error("Training data directory is empty!")
        return False
    return True


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: FullConfig) -> None:
    if not check_required_configurations(cfg):
        exit(-1)
    with open(os.path.join(get_original_cwd(), CONFIG_PATH, CONFIG_NAME), "rb") as config:
        digest = hashlib.sha256(config.read()).hexdigest()
        logging.info(msg=f"Run id: {digest}")

    cfg.paths.mapping_file = str(cfg.paths.mapping_file)
    runner = TrainingRunner(cfg)
    runner.run()


if __name__ == '__main__':
    if not check_config_file_exists():
        inputter.print_error("Could not create config file, aborting!")
        exit(-1)
    main()
