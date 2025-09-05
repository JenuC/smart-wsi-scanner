import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple


class SimpleAcquisitionWorkflow:
    """Hardware-agnostic acquisition workflow."""

    def __init__(self, hardware_interface, config_path: str):
        self.hardware = hardware_interface
        self.config = self._load_config(config_path)
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        """Build processing pipeline from config."""
        steps = []

        if self.config["processing"]["debayer"]:
            steps.append(DebayerStep())

        if self.config["processing"]["background_correction"]["enabled"]:
            steps.append(
                BackgroundCorrectionStep(self.config["processing"]["background_correction"])
            )

        if self.config["processing"]["white_balance"]:
            steps.append(WhiteBalanceStep())

        return ProcessingPipeline(steps)

    def run(self):
        """Execute the acquisition workflow."""
        # 1. Setup
        positions = self._load_positions()
        self._prepare_output_folders()

        # 2. Main acquisition loop
        for pos_idx, (position, filename) in enumerate(positions):
            # Move to position
            self.hardware.move_to_position(position)

            # Autofocus if needed
            if self._should_autofocus(pos_idx):
                self._perform_autofocus()

            # Acquire all angles
            images = {}
            for angle, exposure in zip(
                self.config["imaging"]["angles"], self.config["imaging"]["exposures"]
            ):
                # Hardware-specific angle setting
                self.hardware.set_angle(angle)
                self.hardware.set_exposure(exposure)

                # Acquire and process
                raw_image, metadata = self.hardware.acquire_image()
                processed_image = self.pipeline.process(raw_image, metadata)
                images[angle] = processed_image

                # Save
                self._save_image(processed_image, angle, filename)

            # Optional: Generate derived products (e.g., birefringence)
            self._generate_derived_products(images, filename)
