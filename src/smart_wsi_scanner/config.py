from dataclasses import dataclass, field
from dataclasses import make_dataclass
from typing import Dict, Type, Optional, Any, List
import yaml
import os
from pathlib import Path


## property constraints
@dataclass
class Limits:
    """Stage movement limits"""
    low: float
    high: float

    def __post_init__(self):
        if self.low > self.high:
            self.low, self.high = self.high, self.low

    def __repr__(self):
        return f"Limits(low={self.low:.1f}, high={self.high:.1f})"


@dataclass
class Position:
    """Stage position coordinates"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __repr__(self):
        return f"Position(x={self.x:.1f}, y={self.y:.1f}, z={self.z:.1f})"


## instruments: stage, lens, detector, imaging mode


@dataclass
class StageConfig:
    """Stage movement and limits configuration"""
    x_limits: Limits = field(default_factory=lambda: Limits(-100000, 100000))
    y_limits: Limits = field(default_factory=lambda: Limits(-100000, 100000))
    z_limits: Limits = field(default_factory=lambda: Limits(0, 100000))
    vendor: str = "Unknown"
    home_position: Position = field(default_factory=Position)


@dataclass
class ImagingConfig:
    """Imaging mode configuration"""
    mode_name: str = "default"
    pixel_size: float = 1.0
    magnification: float = 1.0
    na: float = 0.1
    working_distance: Optional[float] = None
    exposure_time: Optional[float] = None
    light_intensity: Optional[float] = None


@dataclass
class LensConfig:
    """Objective lens configuration"""
    name: str = "default"
    magnification: float = 1.0
    na: float = 0.1
    working_distance: Optional[float] = None
    correction: str = "Plan"


@dataclass
class DetectorConfig:
    """Camera/detector configuration"""
    name: str = "default"
    width: int = 2048
    height: int = 2048
    pixel_size: float = 6.5
    bit_depth: int = 16
    vendor: str = "Unknown"


@dataclass
class Config:
    """Unified configuration class for microscope settings
    
    Examples:
        Basic usage with existing YAML files:
        
        >>> from smart_wsi_scanner.config import ConfigManager
        >>> 
        >>> # Initialize the config manager (automatically loads all .yml files)
        >>> config_manager = ConfigManager()
        >>> 
        >>> # List all available configurations
        >>> print("Available configs:", config_manager.list_configs())
        >>> # Output: ['config_PPM', 'config_CAMM']
        >>> 
        >>> # Load a specific configuration
        >>> ppm_config = config_manager.get_config('config_PPM')
        >>> print(ppm_config)
        >>> # Output will show all configuration details including:
        >>> # - Stage limits
        >>> # - Imaging modes
        >>> # - Lens specifications
        >>> # - Detector settings
        >>> 
        >>> # Access specific settings
        >>> print("Stage X limits:", ppm_config.stage.x_limits)
        >>> print("Imaging mode:", ppm_config.imaging.mode_name)
        >>> print("Lens magnification:", ppm_config.lens.magnification)
        >>> 
        >>> # Create a new configuration
        >>> new_config = Config(
        ...     name="custom_config",
        ...     description="Custom microscope settings",
        ...     stage=StageConfig(
        ...         x_limits=Limits(-50000, 50000),
        ...         y_limits=Limits(-50000, 50000),
        ...         z_limits=Limits(0, 10000)
        ...     ),
        ...     imaging=ImagingConfig(
        ...         mode_name="BF_20x",
        ...         pixel_size=0.2271,
        ...         magnification=20.0,
        ...         na=0.4
        ...     )
        ... )
        >>> 
        >>> # Save the new configuration
        >>> config_manager.save_config('custom_config', new_config)
        >>> 
        >>> # Load it back
        >>> loaded_config = config_manager.get_config('custom_config')
        >>> print(loaded_config)
        
        Working with YAML files directly:
        
        >>> import yaml
        >>> 
        >>> # Load YAML file
        >>> with open('config_PPM.yml', 'r') as f:
        ...     yaml_data = yaml.safe_load(f)
        >>> 
        >>> # Convert to Config object
        >>> config = Config.from_yaml(yaml_data)
        >>> 
        >>> # Modify settings
        >>> config.stage.x_limits = Limits(-60000, 60000)
        >>> 
        >>> # Convert back to YAML
        >>> yaml_data = config.to_yaml()
        >>> 
        >>> # Save to file
        >>> with open('modified_config.yml', 'w') as f:
        ...     yaml.dump(yaml_data, f)
    """
    name: str = "default"
    version: str = "1.0"
    description: str = "Default configuration"
    
    stage: StageConfig = field(default_factory=StageConfig)
    imaging: ImagingConfig = field(default_factory=ImagingConfig)
    lens: LensConfig = field(default_factory=LensConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    
    instrument_specific: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_data: Dict[str, Any]) -> 'Config':
        """Create a Config instance from YAML data"""
        # Create default config
        config = cls()
        
        # Update with YAML data if available
        if 'name' in yaml_data:
            config.name = yaml_data['name']
        if 'version' in yaml_data:
            config.version = yaml_data['version']
        if 'description' in yaml_data:
            config.description = yaml_data['description']
            
        # Update stage config
        if 'stage' in yaml_data:
            stage_data = yaml_data['stage']
            if 'xlimit' in stage_data:
                config.stage.x_limits = Limits(**stage_data['xlimit'])
            if 'ylimit' in stage_data:
                config.stage.y_limits = Limits(**stage_data['ylimit'])
            if 'zlimit' in stage_data:
                config.stage.z_limits = Limits(**stage_data['zlimit'])
            if 'vendor' in stage_data:
                config.stage.vendor = stage_data['vendor']
                
        # Update imaging config
        if 'imagingMode' in yaml_data:
            # Take the first imaging mode as default
            mode_name, mode_data = next(iter(yaml_data['imagingMode'].items()))
            config.imaging.mode_name = mode_name
            if 'pixelSize_um' in mode_data:
                config.imaging.pixel_size = mode_data['pixelSize_um']
            if 'objectiveLens' in mode_data:
                config.lens.name = mode_data['objectiveLens']
                
        # Update lens config
        if 'objectiveLens' in yaml_data:
            lens_data = yaml_data['objectiveLens']
            if 'name' in lens_data:
                config.lens.name = lens_data['name']
            if 'magnification' in lens_data:
                config.lens.magnification = lens_data['magnification']
            if 'NA' in lens_data:
                config.lens.na = lens_data['NA']
            if 'WD' in lens_data:
                config.lens.working_distance = lens_data['WD']
                
        # Update detector config
        if 'detector' in yaml_data:
            detector_data = yaml_data['detector']
            if 'name' in detector_data:
                config.detector.name = detector_data['name']
            if 'width' in detector_data:
                config.detector.width = detector_data['width']
            if 'height' in detector_data:
                config.detector.height = detector_data['height']
                
        return config

    def to_yaml(self) -> Dict[str, Any]:
        """Convert Config instance to YAML-compatible dictionary"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'stage': {
                'xlimit': {'low': self.stage.x_limits.low, 'high': self.stage.x_limits.high},
                'ylimit': {'low': self.stage.y_limits.low, 'high': self.stage.y_limits.high},
                'zlimit': {'low': self.stage.z_limits.low, 'high': self.stage.z_limits.high},
                'vendor': self.stage.vendor
            },
            'imaging': {
                'mode_name': self.imaging.mode_name,
                'pixel_size': self.imaging.pixel_size,
                'magnification': self.imaging.magnification,
                'na': self.imaging.na
            },
            'lens': {
                'name': self.lens.name,
                'magnification': self.lens.magnification,
                'na': self.lens.na,
                'working_distance': self.lens.working_distance
            },
            'detector': {
                'name': self.detector.name,
                'width': self.detector.width,
                'height': self.detector.height,
                'pixel_size': self.detector.pixel_size
            }
        }

    def __repr__(self) -> str:
        """Pretty print configuration details"""
        config_details = []
        config_details.append(f"Config(name='{self.name}', version='{self.version}')")
        config_details.append(f"Description: {self.description}")
        
        config_details.append("\nStage Configuration:")
        config_details.append(f"  X Limits: {self.stage.x_limits}")
        config_details.append(f"  Y Limits: {self.stage.y_limits}")
        config_details.append(f"  Z Limits: {self.stage.z_limits}")
        config_details.append(f"  Vendor: {self.stage.vendor}")
        
        config_details.append("\nImaging Configuration:")
        config_details.append(f"  Mode: {self.imaging.mode_name}")
        config_details.append(f"  Pixel Size: {self.imaging.pixel_size}")
        config_details.append(f"  Magnification: {self.imaging.magnification}")
        config_details.append(f"  NA: {self.imaging.na}")
        
        config_details.append("\nLens Configuration:")
        config_details.append(f"  Name: {self.lens.name}")
        config_details.append(f"  Magnification: {self.lens.magnification}")
        config_details.append(f"  NA: {self.lens.na}")
        if self.lens.working_distance:
            config_details.append(f"  Working Distance: {self.lens.working_distance}")
            
        config_details.append("\nDetector Configuration:")
        config_details.append(f"  Name: {self.detector.name}")
        config_details.append(f"  Resolution: {self.detector.width}x{self.detector.height}")
        config_details.append(f"  Pixel Size: {self.detector.pixel_size}")
        config_details.append(f"  Bit Depth: {self.detector.bit_depth}")
        
        return "\n".join(config_details)


class ConfigManager:
    """Manages microscope configurations and presets"""
    
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            # Use the submodule path by default
            package_dir = Path(__file__).parent
            self.config_dir = package_dir / "configurations"
        else:
            self.config_dir = Path(config_dir)
            
        self._configs: Dict[str, Config] = {}
        self._load_configs()
        
    def _load_configs(self) -> None:
        """Load all configuration files from config directory"""
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {self.config_dir}")
            
        for file in self.config_dir.glob("*.yml"):
            config_name = file.stem
            self._configs[config_name] = self.load_config(str(file))
                
    def load_config(self, config_path: str) -> Config:
        """Load a single configuration file"""
        with open(config_path, "r") as file:
            data = yaml.safe_load(file)
        return Config.from_yaml(data)
        
    def get_config(self, name: str) -> Optional[Config]:
        """Get configuration by name"""
        return self._configs.get(name)
        
    def save_config(self, name: str, config: Config) -> None:
        """Save configuration to file"""
        config_path = self.config_dir / f"{name}.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config.to_yaml(), f)
        self._configs[name] = config
        
    def list_configs(self) -> list:
        """List all available configurations"""
        return list(self._configs.keys())

    def __repr__(self) -> str:
        """Pretty print configuration manager details"""
        config_details = []
        config_details.append("ConfigManager:")
        config_details.append(f"  Configuration Directory: {self.config_dir}")
        config_details.append("  Available Configurations:")
        
        for name, config in self._configs.items():
            config_details.append(f"\n  {name}:")
            config_details.append(str(config))
        
        return "\n".join(config_details)
