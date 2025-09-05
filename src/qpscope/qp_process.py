from abc import ABC, abstractmethod
from typing import List
import numpy as np


class ProcessingStep(ABC):
    """Base class for processing steps."""

    @abstractmethod
    def process(self, image: np.ndarray, metadata: dict) -> np.ndarray:
        pass


class ProcessingPipeline:
    """Sequential processing pipeline."""

    def __init__(self, steps: List[ProcessingStep]):
        self.steps = steps

    def process(self, image: np.ndarray, metadata: dict) -> np.ndarray:
        for step in self.steps:
            image = step.process(image, metadata)
        return image


# Example step implementations

class BackgroundCorrectionStep(ProcessingStep):
    def __init__(self, config):
        self.backgrounds = self._load_backgrounds(config)

    def process(self, image, metadata):
        angle = metadata.get("angle")
        if angle in self.backgrounds:
            return apply_flat_field(image, self.backgrounds[angle])
        return image

    def _load_backgrounds(self, config):
        # Load background images based on config
        # returns a dict mapping angles to background images
        return {}


def apply_flat_field(image: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Placeholder for flat-field correction logic."""
    # Implement flat-field correction logic here
    return image  # Replace with actual corrected image
