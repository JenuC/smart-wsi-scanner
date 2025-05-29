from smart_wsi_scanner.smartpath import init_pycromanager
from smart_wsi_scanner.config import ConfigManager, sp_position
from smart_wsi_scanner.hardware import PycromanagerHardware
import argparse
import sys
import numpy as np
from skimage import img_as_ubyte, exposure

core, studio = init_pycromanager()
config_manager = ConfigManager()
if not core:
    print("Failed to initialize Micro-Manager connection")
ppm_settings = config_manager.get_config('config_PPM')
hardware = PycromanagerHardware(core, ppm_settings, studio)
brushless = "KBD101_Thor_Rotation"
current_position_xyz = hardware.get_current_position()

from multiprocessing import Process
from tkinter import Tk, Label, Button, Frame
from PIL import Image, ImageTk

def image_window():
    # Create a process for the Tkinter window
    p = Process(target=run_tk_image_window)
    p.start()    
    return p

def snap_with_preview():
    viewer = image_window()
    viewer.join()
    
def snap_image():
    image, metadata = hardware.snap_image()
    return image

def process_image(image_array):
    """Process image for display - converting from various formats to PIL Image"""
    if image_array is None:
        return None
        
    try:
        if len(image_array.shape) == 2:
            # Grayscale image
            if image_array.dtype == np.uint16 or image_array.dtype == np.int16:
                # Convert 16-bit to 8-bit using skimage's rescaling
                image_array = img_as_ubyte(exposure.rescale_intensity(image_array))
            image = Image.fromarray(image_array.astype('uint8'), 'L')
        else:
            # RGB image (assuming 3D array with RGB channels)
            try:
                if image_array.dtype == np.uint16 or image_array.dtype == np.int16:
                    # Convert 16-bit RGB to 8-bit RGB
                    image_array = img_as_ubyte(exposure.rescale_intensity(image_array))
                image = Image.fromarray(image_array.astype('uint8'), 'RGB')
            except ValueError:
                print("Error reading image")
                return None
                
        if image.size[0] > 1000:
            image = image.resize((270, 200), Image.LANCZOS)
        
        return image
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None
  
def run_tk_image_window():
    """Create a Tkinter window with an image display and snap button"""
    root = Tk()
    root.title("QP-test")
    
    # Create frame for the image
    image_frame = Frame(root, width=270, height=200, bg="grey")
    image_frame.pack(pady=10)
    
    # Label to hold the image
    image_label = Label(image_frame)
    image_label.pack()
    
    # Variable to hold the PhotoImage reference
    photo_image = None
    
    def update_image():
        """Snap a new image and update the display"""
        nonlocal photo_image
        
        # Snap new image
        image_array = snap_image()
        
        # Process image
        pil_image = process_image(image_array)
        
        if pil_image:
            # Update the displayed image
            photo_image = ImageTk.PhotoImage(pil_image)
            image_label.config(image=photo_image)
            snap_button.config(text="Snap Again")
    
    # Button to snap image
    snap_button = Button(root, text="Snap Image", command=update_image)
    snap_button.pack(pady=10)
    
    # Close button
    close_button = Button(root, text="Close", command=root.destroy)
    close_button.pack(pady=5)
    
    # Run the Tkinter event loop
    root.mainloop()


    
def main():
    """Launch the modern UI interface"""
    #from smart_wsi_scanner.modern_snap_ui import main
    #from smart_wsi_scanner.customtkinter_snap_ui import main
    from smart_wsi_scanner.ttkbootstrap_snap_ui import main as control
    control()
    #snap_with_preview()
    
#TODO : quit the main and break multiprocess for Tk
#TODO: move to custom-tk
if __name__=='__main__':
    # Uncomment the line below to use modern UI
    main()
