from dataclasses import dataclass, field
from dataclasses import make_dataclass
from typing import Dict, Type, Optional
import yaml
import os
from pathlib import Path


class sp:
    def __init__(self) -> None:
        self.microscope_settings = sp_microscope_settings
        self.position = sp_position
        self.imaging_mode = sp_imaging_mode
        self.detector = sp_detector
        self.stage = sp_stage_settings
        self.objective_lens = sp_objective_lens
        self.camm_settings = sp_camm_settings
        self.ppm_settings = sp_ppm_settings
        self.limits = _limits


## property constraints
@dataclass
class _limits:
    low: float
    high: float

    def __post_init__(self):
        if self.low > self.high:
            self.low, self.high = self.high, self.low


@dataclass
class sp_position:
    x: Optional[float] = field(default=None)
    y: Optional[float] = field(default=None)
    z: Optional[float] = field(default=None)

    def __post_init__(self):
        if self.x is not None and not isinstance(self.x, (float, int)):
            print(
                f"Invalid type for x: expected float or int, got {type(self.x).__name__} ({self.x!r})"
            )
        if self.y is not None and not isinstance(self.y, (float, int)):
            print(
                f"Invalid type for y: expected float or int, got {type(self.y).__name__} ({self.y!r})"
            )
        if self.z is not None and not isinstance(self.z, (float, int)):
            print(
                f"Invalid type for z: expected float or int, got {type(self.z).__name__} ({self.z!r})"
            )

    def populate_missing(self, current_position: "sp_position") -> None:
        """Populate missing coordinates with values from current_position."""
        if self.x is None:
            self.x = current_position.x
        if self.y is None:
            self.y = current_position.y
        if self.z is None:
            self.z = current_position.z

    def __repr__(self):
        kws_values = [
            f"{key}={value:.1f}" for key, value in self.__dict__.items() if value is not None
        ]
        kws_none = [f"{key}={value!r}" for key, value in self.__dict__.items() if value is None]
        kws = kws_values + kws_none
        return f"{type(self).__name__}({', '.join(kws)})"


## instruments: stage, lens, detector, imaging mode


@dataclass
class sp_stage_settings:
    x_limit: Optional[_limits] = field(default=None)
    y_limit: Optional[_limits] = field(default=None)
    z_limit: Optional[_limits] = field(default=None)


@dataclass
class sp_objective_lens:
    name: str
    magnification: float
    NA: float
    WD: Optional[float] = field(default=None)


@dataclass
class sp_detector:
    width: Optional[int] = field(default=None)
    height: Optional[int] = field(default=None)


@dataclass
class sp_imaging_mode:
    name: Optional[str] = field(default=None)
    pixel_size: Optional[float] = field(default=None)


## microscope settings


@dataclass
class sp_microscope:
    name: Optional[str] = field(default=None)
    type: Optional[str] = field(default=None)


@dataclass
class sp_microscope_settings:
    path: Optional[str] = field(default=None)
    microscope: Optional[sp_microscope] = field(default=None)
    stage: Optional[sp_stage_settings] = field(default=None)
    lens: Optional[sp_objective_lens] = field(default=None)
    detector: Optional[sp_detector] = field(default=None)
    imaging_mode: Optional[sp_imaging_mode] = field(default=None)


## instrument specific adaptation


@dataclass
class sp_camm_settings(sp_microscope_settings):
    slide_size: Optional[sp_objective_lens] = field(default=None)
    lamp: Optional[sp_stage_settings] = field(default=None)
    objective_slider: Optional[sp_detector] = field(default=None)


class sp_ppm_settings(sp_microscope_settings):
    slide_size: Optional[sp_objective_lens] = field(default=None)


## YAML support


def read_yaml_file(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")
    with open(filename, "r") as file:
        data = yaml.safe_load(file)
    return data


def create_dataclass(name, data):
    fields = []
    for key, value in data.items():
        if isinstance(value, dict):
            # Recursively create nested data classes for nested dictionaries
            # print(value)
            nested_class = create_dataclass(key.capitalize(), value)
            fields.append((key, nested_class))
        else:
            fields.append((key, type(value)))
    DataClass = make_dataclass(name, fields)
    # print(DataClass)
    return DataClass


def instantiate_dataclass(data_class, data):
    kwargs = {}
    for fieldx in data_class.__dataclass_fields__:
        value = data[fieldx]
        field_type = data_class.__dataclass_fields__[fieldx].type
        if isinstance(value, dict):
            value = instantiate_dataclass(field_type, value)
        kwargs[fieldx] = value
    return data_class(**kwargs)


def yaml_to_dataclass(yaml_data):
    DataClass = create_dataclass("DataClass", yaml_data)
    instance = instantiate_dataclass(DataClass, yaml_data)
    return instance


class ConfigManager:
    """Manages microscope configurations and presets"""

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            # Use the submodule path by default
            package_dir = Path(__file__).parent
            self.config_dir = package_dir / "configurations"
        else:
            self.config_dir = Path(config_dir)

        self._configs: Dict[str, Type[sp_microscope_settings]] = {}
        self._load_configs()

    def _load_configs(self) -> None:
        """Load all configuration files from config directory"""
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {self.config_dir}")

        for file in self.config_dir.glob("*.yml"):
            config_name = file.stem
            self._configs[config_name] = self.load_config(str(file))

    def load_config(self, config_path: str) -> Type[sp_microscope_settings]:
        """Load a single configuration file"""
        data = read_yaml_file(config_path)
        return yaml_to_dataclass(data)

    def get_config(self, name: str) -> Optional[Type[sp_microscope_settings]]:
        """Get configuration by name"""
        return self._configs.get(name)

    def save_config(self, name: str, config: sp_microscope_settings) -> None:
        """Save configuration to file"""
        config_path = self.config_dir / f"{name}.yml"
        with open(config_path, "w") as f:
            yaml.dump(config.__dict__, f)
        self._configs[name] = config

    def list_configs(self) -> list:
        """List all available configurations"""
        return list(self._configs.keys())
