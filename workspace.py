import logging
import os
import time

import omegaconf

logger = logging.getLogger(__name__)


def load_config(config_path, command_line_args=None):

    logger.info(f"Loading from: {config_path}")

    config = omegaconf.OmegaConf.load(config_path)

    if hasattr(config, "include"):
        base_config_paths = [os.path.join(os.path.dirname(config_path), include_path) for include_path in config.include]
        base_configs = [load_config(base_config_path) for base_config_path in base_config_paths]
        config = omegaconf.OmegaConf.merge(*base_configs, config)

    if command_line_args is not None:
        command_line_config = omegaconf.OmegaConf.from_dotlist(command_line_args)
        config = omegaconf.OmegaConf.merge(config, command_line_config)

    return config


def create_workspace(config):

    config.name = time.strftime(config.name, time.localtime())

    os.makedirs(config.save_dir)

    omegaconf.OmegaConf.save(config, os.path.join(config.save_dir, "config.yaml"))

    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(config.save_dir, "output.log")),
            logging.StreamHandler(),
        ],
    )

    return config
