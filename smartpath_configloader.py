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
            # Recursively nested data classes for nested dictionaries
            nested_class = create_dataclass(key.capitalize(), value)
            fields.append((key, nested_class))
        else:
            fields.append((key, type(value)))
    #print(name,fields)
    DataClass = make_dataclass(name, fields)
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

def load_config(configfile):    
    return yaml_to_dataclass(read_yaml_file(configfile))

#camm = yaml_to_dataclass(read_yaml_file("./config/config_CAMM.yml"))
#ppm = yaml_to_dataclass(read_yaml_file("./config/config_PPM.yml"))
#rsc = yaml_to_dataclass(read_yaml_file("./config/resources_LOCI.yml"))
        
