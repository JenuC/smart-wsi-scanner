#!/usr/bin/env python
"""Minimal DearPyGui example to demonstrate proper usage."""

import dearpygui.dearpygui as dpg
import numpy as np

def create_dummy_image():
    """Create a simple dummy image for demonstration."""
    # Create a 100x100 grayscale image with a gradient
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # Create a simple pattern
    Z = np.sin(X) * np.cos(Y) * 0.5 + 0.5
    Z = Z * 255  # Scale to 0-255 range
    
    # Add a circle in the center
    center_x, center_y = 50, 50
    circle = (X - center_x)**2 + (Y - center_y)**2 < 20**2
    Z[circle] = 255
    
    return Z.astype(np.uint8)

def main():
    """Create and run a minimal DearPyGui application."""
    # Create context and viewport
    dpg.create_context()
    dpg.create_viewport(title="DearPyGui Example", width=800, height=600)
    
    # Create a window
    with dpg.window(label="Example Window", width=800, height=600):
        # Add some basic controls
        dpg.add_text("Hello, DearPyGui!")
        dpg.add_button(label="Click Me", callback=lambda: print("Button clicked!"))
        
        # Add a slider
        dpg.add_slider_float(label="Slider", default_value=0.5, min_value=0.0, max_value=1.0)
        
        # Add a plot
        with dpg.plot(label="Example Plot", height=300, width=700):
            dpg.add_plot_legend()
            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="X")
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Y")
            
            # Create a line series
            x_data = np.linspace(0, 10, 100)
            y_data = np.sin(x_data)
            dpg.add_line_series(x=x_data.tolist(), y=y_data.tolist(), label="Sine Wave", parent=y_axis)
        
        # Add an image display
        dpg.add_text("Image Display")
        with dpg.plot(label="Image", height=300, width=700):
            dpg.add_plot_legend()
            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="X")
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Y")
            
            # Create a heatmap series for the image
            image = create_dummy_image()
            rows, cols = image.shape
            dpg.add_heat_series(
                rows=rows,
                cols=cols,
                values=image.tolist(),
                label="Image",
                tag="image_heatmap",
                parent=y_axis
            )
    
    # Setup and run
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main() 