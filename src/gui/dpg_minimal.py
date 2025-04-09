#!/usr/bin/env python
"""Very minimal DearPyGui example."""

import dearpygui.dearpygui as dpg

def main():
    """Create and run a minimal DearPyGui application."""
    # Create context and viewport
    dpg.create_context()
    dpg.create_viewport(title="Minimal DearPyGui Example", width=400, height=300)
    
    # Create a window
    with dpg.window(label="Example Window", width=400, height=300):
        # Add some basic controls
        dpg.add_text("Hello, DearPyGui!")
        
        # Add a button with a callback
        def button_callback():
            print("Button clicked!")
            dpg.set_value("text_value", "Button was clicked!")
        
        dpg.add_button(label="Click Me", callback=button_callback)
        
        # Add a slider
        dpg.add_slider_float(label="Slider", default_value=0.5, min_value=0.0, max_value=1.0)
        
        # Add a text field that can be updated
        dpg.add_text("This text will be updated", tag="text_value")
    
    # Setup and run
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main() 