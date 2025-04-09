"""GUI module for microscope control with hardware abstraction."""

import dearpygui.dearpygui as dpg
import numpy as np
from typing import Optional, Tuple, Dict, Any
from ..smart_wsi_scanner.hardware import MicroscopeHardware, PycromanagerHardware
from ..smart_wsi_scanner.config import sp_position, sp_microscope_settings, ConfigManager

class MicroscopeGUI:
    """GUI class for microscope control with hardware abstraction."""
    
    def __init__(self, hardware: Optional[MicroscopeHardware] = None):
        """Initialize the GUI with optional hardware."""
        self.hardware = hardware
        self.image_data = None
        self.image_metadata = None
        
        # Create a dummy image for testing when hardware is not available
        self._create_dummy_image()
        
        # Initialize GUI
        self._setup_gui()
        
    def _create_dummy_image(self):
        """Create a dummy image for testing when hardware is not available."""
        # Create a 512x512 grayscale image with a gradient
        x = np.linspace(0, 10, 512)
        y = np.linspace(0, 10, 512)
        X, Y = np.meshgrid(x, y)
        
        # Create a pattern with some circles and gradients
        Z = np.sin(X) * np.cos(Y) * 0.5 + 0.5
        Z = Z * 255  # Scale to 0-255 range
        
        # Add some circles
        center_x, center_y = 256, 256
        for r in range(50, 200, 50):
            circle = (X - center_x)**2 + (Y - center_y)**2 < r**2
            Z[circle] = 255 - r/2
        
        # Convert to uint8
        self.dummy_image = Z.astype(np.uint8)
        
    def _setup_gui(self):
        """Set up the Dear PyGui interface."""
        dpg.create_context()
        dpg.create_viewport(title='Smart WSI Scanner', width=800, height=600)
        
        # Create main window
        with dpg.window(label="Microscope Control", width=800, height=600):
            # Create layout
            with dpg.group(horizontal=True):
                # Left panel for controls
                with dpg.group(width=200):
                    dpg.add_text("Controls")
                    dpg.add_separator()
                    
                    # Position control
                    dpg.add_text("Position Control")
                    dpg.add_input_float(label="X", default_value=0.0, tag="pos_x")
                    dpg.add_input_float(label="Y", default_value=0.0, tag="pos_y")
                    dpg.add_input_float(label="Z", default_value=0.0, tag="pos_z")
                    dpg.add_button(label="Move to Position", callback=self._move_to_position)
                    
                    dpg.add_separator()
                    
                    # Objective control
                    dpg.add_text("Objective Control")
                    dpg.add_combo(
                        label="Objective",
                        items=["4X", "20X"],
                        default_value="4X",
                        tag="objective"
                    )
                    dpg.add_button(label="Set Objective", callback=self._set_objective)
                    
                    dpg.add_separator()
                    
                    # Image capture
                    dpg.add_text("Image Capture")
                    dpg.add_button(label="Snap Image", callback=self._snap_image)
                    dpg.add_button(label="Autofocus", callback=self._autofocus)
                    
                    # Add a button to load dummy image
                    dpg.add_separator()
                    dpg.add_button(label="Load Dummy Image", callback=self._load_dummy_image)
                
                # Right panel for image display
                with dpg.group(width=580):
                    dpg.add_text("Image Preview")
                    with dpg.plot(label="Image", height=400, width=560):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="X")
                        dpg.add_plot_axis(dpg.mvYAxis, label="Y")
                        # Use scatter plot instead of plot_series
                        dpg.add_scatter_series([], [], label="Image", tag="image_series")
    
    def _load_dummy_image(self):
        """Load the dummy image for testing."""
        self.image_data = self.dummy_image
        self._update_image_display()
        print("Loaded dummy image")
    
    def _move_to_position(self):
        """Move to the specified position."""
        if self.hardware is None:
            print("No hardware connected")
            return
            
        x = dpg.get_value("pos_x")
        y = dpg.get_value("pos_y")
        z = dpg.get_value("pos_z")
        
        position = sp_position(x=x, y=y, z=z)
        try:
            self.hardware.move_to_position(position)
            print(f"Moved to position: {position}")
        except Exception as e:
            print(f"Error moving to position: {e}")
    
    def _set_objective(self):
        """Set the objective lens."""
        if self.hardware is None:
            print("No hardware connected")
            return
            
        objective = dpg.get_value("objective")
        try:
            self.hardware.set_objective(objective)
            print(f"Set objective to: {objective}")
        except Exception as e:
            print(f"Error setting objective: {e}")
    
    def _snap_image(self):
        """Capture an image."""
        if self.hardware is None:
            print("No hardware connected")
            return
            
        try:
            self.image_data, self.image_metadata = self.hardware.snap_image()
            self._update_image_display()
            print("Image captured successfully")
        except Exception as e:
            print(f"Error capturing image: {e}")
    
    def _autofocus(self):
        """Perform autofocus."""
        if self.hardware is None:
            print("No hardware connected")
            return
            
        try:
            best_focus = self.hardware.autofocus()
            dpg.set_value("pos_z", best_focus)
            print(f"Autofocus completed. Best focus: {best_focus}")
        except Exception as e:
            print(f"Error during autofocus: {e}")
    
    def _update_image_display(self):
        """Update the image display in the GUI."""
        if self.image_data is None:
            return
            
        # Convert image data to format suitable for display
        if len(self.image_data.shape) == 3:  # Color image
            image = self.image_data
        else:  # Grayscale image
            image = np.stack([self.image_data] * 3, axis=-1)
            
        # Create scatter plot data from image
        height, width = image.shape[:2]
        x_data = np.arange(width)
        y_data = np.arange(height)
        X, Y = np.meshgrid(x_data, y_data)
        
        # Flatten the arrays for scatter plot
        x_flat = X.flatten()
        y_flat = Y.flatten()
        
        # Get pixel values for color
        if len(image.shape) == 3:
            # For color images, use the first channel for intensity
            z_flat = image[:,:,0].flatten()
        else:
            z_flat = image.flatten()
        
        # Update the scatter plot
        dpg.set_value("image_series", [x_flat.tolist(), y_flat.tolist()])
    
    def run(self):
        """Run the GUI application."""
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

def create_gui_with_hardware():
    """Create GUI with hardware if available."""
    try:
        from ..smart_wsi_scanner.smartpath import smartpath
        core = smartpath.core
        if core is None:
            raise ValueError("Core is not initialized")
        config_manager = ConfigManager()
        settings = config_manager.get_config('config_CAMM')
        if settings is None:
            raise ValueError("Settings are not initialized")
        hardware = PycromanagerHardware(core, settings)
        print("Hardware initialized successfully")
    except Exception as e:
        print(f"Could not initialize hardware: {e}")
        hardware = None
    
    return MicroscopeGUI(hardware)

if __name__ == "__main__":
    # Create and run GUI
    gui = create_gui_with_hardware()
    gui.run() 