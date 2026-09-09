import logging
import os
import time

import omegaconf

logger = logging.getLogger(__name__)


def load_config(config_path, command_line_args=None):
    """Loads the config file using OmegaConf, performing merges with base configs and the command line arguments"""

    logger.info(f"Loading from: {config_path}")

    # Load the config using OmegaConf
    config = omegaconf.OmegaConf.load(config_path)

    # Load all the base configs to include, and merge them with the current config, giving precedence to the current config
    if hasattr(config, "include"):
        base_config_paths = [os.path.join(os.path.dirname(config_path), include_path) for include_path in config.include]
        base_configs = [load_config(base_config_path) for base_config_path in base_config_paths]
        config = omegaconf.OmegaConf.merge(*base_configs, config)

    # Load the command line arguments, and merge them with the current config, giving precedence to the command line
    if command_line_args is not None:
        command_line_config = omegaconf.OmegaConf.from_dotlist(command_line_args)
        config = omegaconf.OmegaConf.merge(config, command_line_config)

    return config


def create_workspace(config):
    """Create a results folder in the target directory"""

    # Treat the name as a time.strftime format string (so that every experiment is named after when it was run)
    config.name = time.strftime(config.name, time.localtime())

    # Create the results directory
    os.makedirs(config.save_dir)

    # Save the config to the results directory
    omegaconf.OmegaConf.save(config, os.path.join(config.save_dir, "config.yaml"))

    # Set up the print loggers by removing all handlers associated with the root logger object,
    # then setting up the logger to print messages *and* save them to a file
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
