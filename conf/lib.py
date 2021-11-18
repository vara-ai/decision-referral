import datetime
import os
from typing import Callable, Optional
import uuid

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from hydra.core.config_store import ConfigStore

from conf import git_util


CONFIG_STORE = ConfigStore.instance()  # Hydra ConfigStore singleton


def register_config(name: Optional[str] = None, group: Optional[str] = None) -> Callable:
    """Decorator for registering structured configs (dataclasses) with Hydra"""

    def decorator_config_class(config_class):
        CONFIG_STORE.store(name=config_class.__name__.lower() if name is None else name, node=config_class, group=group)
        return config_class

    return decorator_config_class


def save_config(save_dir: str, config: DictConfig) -> str:
    """Saves a single config to a specified output directory

    Args:
        save_dir: The directory in which to save the config as YAML.
        config: A config object to save as YAML.

    Returns:
        destination path
    """
    filename_parts = [
        "config",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        uuid.uuid4().hex[:4],
    ]
    filename = "-".join(filename_parts) + ".yaml"
    dest_path = os.path.join(save_dir, filename)

    to_save = OmegaConf.merge({"commit_sha": git_util.get_commit_sha()}, config)

    with open(dest_path, "w") as h:
        h.write(OmegaConf.to_yaml(to_save, sort_keys=False, resolve=True))

    return dest_path
