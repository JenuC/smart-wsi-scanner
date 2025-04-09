import numpy as np
import dearpygui.dearpygui as dpg

def main():
    """
    Creates a random RGBA image using NumPy, registers it as a static texture with Dear PyGui,
    and displays it in a simple window.
    """
    # Always create the Dear PyGui context first.
    dpg.create_context()

    # Set image dimensions.
    width, height = 256, 256

    # Create random image data (RGBA; float values between 0 and 1) and flatten it.
    image_data = np.random.rand(height, width, 4).astype(np.float32).flatten()

    # Register the texture in a texture registry.
    with dpg.texture_registry():
        dpg.add_static_texture(width, height, image_data, tag="random_texture")

    # Create a window and add an image widget to display the texture.
    with dpg.window(label="Random NumPy Image", width=width, height=height):
        dpg.add_image("random_texture")

    # Create and configure the viewport.
    dpg.create_viewport(title='Random NumPy Image', width=width, height=height)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Start the Dear PyGui event loop.
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == '__main__':
    main()
