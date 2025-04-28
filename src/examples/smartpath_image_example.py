"""Example of acquiring images using smartpath."""

from pathlib import Path
import matplotlib.pyplot as plt
from smart_wsi_scanner.smartpath import smartpath, init_pycromanager
from smart_wsi_scanner.config import ConfigManager, sp_position
from smart_wsi_scanner.hardware import PycromanagerHardware

def main():
    # Initialize Micro-Manager connection
    core, studio = init_pycromanager()
    if not core:
        print("Failed to initialize Micro-Manager connection")
        return

    # Initialize configuration
    config_manager = ConfigManager()
    camm_settings = config_manager.load_config("config_CAMM.yml")

    # Create hardware instance
    hardware = PycromanagerHardware(core, camm_settings, studio)

    # Initialize smartpath
    sp = smartpath(core)

    # Example 1: Simple image capture
    print("\nExample 1: Simple image capture")
    image, metadata = hardware.snap_image()
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print("Metadata keys:", list(metadata.keys()))

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title("Simple Image Capture")
    plt.axis('off')
    plt.show()

    # Example 2: Image with white balance
    print("\nExample 2: Image with white balance")
    image_wb = sp.white_balance(image)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_wb)
    plt.title("Image with White Balance")
    plt.axis('off')
    plt.show()

    # Example 3: Move to position and capture
    print("\nExample 3: Move to position and capture")
    target_position = sp_position(x=1000.0, y=1000.0, z=0.0)
    hardware.move_to_position(target_position)
    
    # Capture image at new position
    image_at_pos, _ = hardware.snap_image()
    plt.figure(figsize=(10, 10))
    plt.imshow(image_at_pos)
    plt.title("Image at Position (1000, 1000)")
    plt.axis('off')
    plt.show()

    # Example 4: Autofocus and capture
    print("\nExample 4: Autofocus and capture")
    focus_position = hardware.autofocus()
    print(f"Best focus position: {focus_position}")
    
    # Capture image after autofocus
    image_focused, _ = hardware.snap_image()
    plt.figure(figsize=(10, 10))
    plt.imshow(image_focused)
    plt.title("Image After Autofocus")
    plt.axis('off')
    plt.show()

    # Example 5: Save image with metadata
    print("\nExample 5: Save image with metadata")
    output_path = Path("output_images")
    output_path.mkdir(exist_ok=True)
    
    # Save the focused image
    from smart_wsi_scanner.qupath import QuPathScanner
    QuPathScanner.save_image(
        filename=output_path / "focused_image.tif",
        pixel_size_um=camm_settings.imaging_mode.pixelsize,
        data=image_focused
    )
    print(f"Image saved to: {output_path / 'focused_image.tif'}")

if __name__ == "__main__":
    main() 