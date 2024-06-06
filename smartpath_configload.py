from dataclasses import make_dataclass
import yaml


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
    for field in data_class.__dataclass_fields__:
        value = data[field]
        field_type = data_class.__dataclass_fields__[field].type
        if isinstance(value, dict):
            value = instantiate_dataclass(field_type, value)
        kwargs[field] = value
    return data_class(**kwargs)


def yaml_to_dataclass(yaml_data):
    DataClass = create_dataclass("DataClass", yaml_data)
    instance = instantiate_dataclass(DataClass, yaml_data)
    return instance


yaml_data = read_yaml_file("schema_CAMM.yml")
camm_stage = yaml_to_dataclass(yaml_data)
# print(camm_stage)
# print(camm_stage.Stage.ylimit.low)
print(*camm_stage.obj_slider)
