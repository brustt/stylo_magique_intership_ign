from dataclasses import dataclass, field
from typing import Any

from commons.config import CONFIG_PATH
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from src.models.segment_any_change.config_run import ExperimentParams


@dataclass
class MySQLConfig:
    host: str = "localhost"
    port: int = 3306


@dataclass
class UserInterface:
    title: str = "My app"
    width: int = 1024
    height: int = 768


@dataclass
class MyConfig:
    db: MySQLConfig = field(default_factory=MySQLConfig)
    # ui: UserInterface = field(default_factory=UserInterface)
    rand: int = 10


cs = ConfigStore.instance()
# cs.store(name="my_conf", node=MyConfig)

cs.store(name="config", node=ExperimentParams)


@hydra.main(version_base=None, config_name="config", config_path="../config")
def my_app(cfg: Any) -> None:
    # print(f"host={cfg.my_conf.db.host}, port={cfg.my_conf.db.port}, int : {cfg.rand}")
    print(f" params : {cfg}")


if __name__ == "__main__":
    my_app()
