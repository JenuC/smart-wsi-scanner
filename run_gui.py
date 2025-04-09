#!/usr/bin/env python
"""Entry point script for running the Smart WSI Scanner GUI."""

from src.gui.microscope_gui import create_gui_with_hardware

if __name__ == "__main__":
    gui = create_gui_with_hardware()
    gui.run() 