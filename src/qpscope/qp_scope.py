from abc import ABC, abstractmethod
import pymmcore
from pycromanager import Core
from typing import Tuple, Optional
import numpy as np


class Hardware(ABC):
    @abstractmethod
    def snap(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_position(self) -> Tuple[float, float, float]:
        pass

    @abstractmethod
    def set_position(self, x: float = None, y: float = None, z: float = None):
        pass


class PymmcoreHardware(Hardware):
    def __init__(self, config_file: str):
        self.mmc = pymmcore.CMMCore()
        self.mmc.loadSystemConfiguration(config_file)

    def snap(self) -> np.ndarray:
        self.mmc.snapImage()
        return self.mmc.getImage()

    def get_position(self) -> Tuple[float, float, float]:
        return (self.mmc.getXPosition(), self.mmc.getYPosition(), self.mmc.getPosition())

    def set_position(self, x: float = None, y: float = None, z: float = None):
        if x is not None and y is not None:
            self.mmc.setXYPosition(x, y)
        if z is not None:
            self.mmc.setPosition(z)


class PycromanagerHardware(Hardware):
    def __init__(self):
        self.core = Core()

    def snap(self) -> np.ndarray:
        self.core.snap_image()
        return self.core.get_tagged_image().pix

    def get_position(self) -> Tuple[float, float, float]:
        return self.core.get_x_position(), self.core.get_y_position(), self.core.get_position()

    def set_position(self, x: float = None, y: float = None, z: float = None):
        if x is not None and y is not None:
            self.core.set_xy_position(x, y)
        if z is not None:
            self.core.set_position(z)


class Microscope:
    def __init__(self, hardware: Hardware):
        self.hardware = hardware

    def snap(self) -> np.ndarray:
        return self.hardware.snap()

    def move_stage(
        self, x: Optional[float] = None, y: Optional[float] = None, z: Optional[float] = None
    ):
        self.hardware.set_position(x, y, z)

    def get_position(self) -> Tuple[float, float, float]:
        return self.hardware.get_position()
