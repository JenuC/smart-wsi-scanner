from dataclasses import dataclass, field
from dataclasses import make_dataclass
import yaml


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
    x: float
    y: float
    z: float = field(default=None)

    def __post_init__(self):
        if not isinstance(self.x, (float, int)):
            print("X WRONG")

    def __repr__(self):
        kws_values = [
            f"{key}={value:.1f}" for key, value in self.__dict__.items() if value
        ]
        kws_none = [
            f"{key}={value!r}" for key, value in self.__dict__.items() if not value
        ]
        kws = kws_values + kws_none
        return f"{type(self).__name__}({', '.join(kws)})"


## instruments: stage, lens, detector, imaging mode


@dataclass
class sp_stage_settings:
    xlimit: _limits = field(default=None)
    ylimit: _limits = field(default=None)
    zlimit: _limits = field(default=None)


@dataclass
class sp_objective_lens:
    name: str
    magnification: float
    NA: float
    WD: float = field(default=None)


@dataclass
class sp_detector:
    width: int = field(default=None)
    height: int = field(default=None)


@dataclass
class sp_imaging_mode:
    name: str = field(default=None)
    pixelsize: float = field(default=None)


## microscope settings


@dataclass
class sp_microscope_settings:
    stage: sp_stage_settings = field(default=None)
    lens: sp_objective_lens = field(default=None)
    detector: sp_detector = field(default=None)
    imaging_mode: sp_imaging_mode = field(default=None)


## instrument specific adaptation


@dataclass
class sp_camm_settings(sp_microscope_settings):
    slide_size: sp_objective_lens = field(default=None)
    lamp: sp_stage_settings = field(default=None)
    objective_slider: sp_detector = field(default=None)


class sp_ppm_settings(sp_microscope_settings):
    slide_size: sp_objective_lens = field(default=None)


## YAML support


def read_yaml_file(filename):
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
