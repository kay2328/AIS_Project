from pathlib import Path
import yaml
from enum import Enum

# current projects. 
class Project(Enum):
    HAZARD_DETECTION = "hazard_detection/project_config.yaml"
    SCENE_DESCRIPTION = "image_description/project_config.yaml"
    
class Config:
    """
    Initialise from a file path. Use ConfigFactory instead.
    """
    def __init__(self, config_path: Path):
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def dict(self):
        return self.config


class ConfigFactory:
    """
    Use Project enum or custom file path from project root.
    """
    BASE_DIR = Path(__file__).parent.parent.parent

    @staticmethod
    def get_config(project: Project) -> Config:
        config_path = ConfigFactory.BASE_DIR / project.value
        return Config(config_path)

    @staticmethod
    def get_config_file(path: Path) -> Config:
        config_path = ConfigFactory.BASE_DIR / path
        return Config(path)