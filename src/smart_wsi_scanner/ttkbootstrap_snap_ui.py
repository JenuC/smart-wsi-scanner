import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import threading
import time
from PIL import Image, ImageTk
import numpy as np
from skimage import img_as_ubyte, exposure

# Import the hardware functions from qp_snap
from smart_wsi_scanner.qp_snap import snap_image, process_image

class ModernMicroscopeViewer:
    def __init__(self):
        self.root = ttk.Window(themename="darkly")
        self.root.title("Smart WSI Scanner")
        self.root.geometry("400x600")
        self.root.minsize(400, 600)
        
        # Variables
        self.photo_image = None
        self.live_view_active = False
        self.live_thread = None
        self.stop_event = threading.Event()
        self.refresh_rate = tk.IntVar(value=1000)
        
        self._create_widgets()
        self._create_layout()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_widgets(self):
        # Main container
        self.main_frame = ttk.Frame(self.root, padding=10)
        
        # Title
        self.title_label = ttk.Label(
            self.main_frame,
            text="Smart WSI Scanner",
            font=("Helvetica", 16, "bold")
        )
        
        # Image display
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_canvas = tk.Canvas(
            self.image_frame,
            width=350,
            height=250,
            bg="#1E1E1E",
            highlightthickness=1,
            highlightbackground="#0D6EFD"
        )
        
        # Status bar
        self.status_label = ttk.Label(
            self.main_frame,
            text="Ready",
            font=("Helvetica", 10)
        )
        
        # Control buttons
        self.button_frame = ttk.Frame(self.main_frame)
        self.snap_button = ttk.Button(
            self.button_frame,
            text="Snap Image",
            command=self.update_image,
            style="primary.TButton",
            width=15
        )
        self.live_button = ttk.Button(
            self.button_frame,
            text="Start Live View",
            command=self.toggle_live_view,
            width=15
        )
        self.save_button = ttk.Button(
            self.button_frame,
            text="Save Image",
            command=self.save_image,
            width=15
        )
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.main_frame,
            orient="horizontal",
            length=350,
            mode="indeterminate"
        )
        
        # Settings frame
        self.settings_frame = ttk.LabelFrame(
            self.main_frame,
            text="Settings",
            padding=10
        )
        
        # Refresh rate control
        self.refresh_label = ttk.Label(
            self.settings_frame,
            text="Refresh Rate (ms):"
        )
        self.refresh_slider = ttk.Scale(
            self.settings_frame,
            from_=100,
            to=2000,
            variable=self.refresh_rate,
            orient="horizontal",
            length=250
        )
        self.refresh_value = ttk.Label(
            self.settings_frame,
            textvariable=self.refresh_rate
        )
        
        # Version label
        self.version_label = ttk.Label(
            self.main_frame,
            text="v1.0.0",
            font=("Helvetica", 8)
        )
    
    def _create_layout(self):
        # Main frame
        self.main_frame.pack(fill=BOTH, expand=YES)
        
        # Title
        self.title_label.pack(pady=(0, 10))
        
        # Image frame
        self.image_frame.pack(fill=BOTH, expand=YES, pady=10)
        self.image_canvas.pack(fill=BOTH, expand=YES)
        
        # Status
        self.status_label.pack(fill=X, pady=5)
        
        # Buttons
        self.button_frame.pack(fill=X, pady=10)
        self.snap_button.pack(side=LEFT, padx=5)
        self.live_button.pack(side=LEFT, padx=5)
        self.save_button.pack(side=LEFT, padx=5)
        
        # Progress bar
        self.progress.pack(fill=X, pady=5)
        
        # Settings
        self.settings_frame.pack(fill=X, pady=10)
        self.refresh_label.pack(anchor=W)
        self.refresh_slider.pack(fill=X, pady=5)
        self.refresh_value.pack(anchor=E)
        
        # Version
        self.version_label.pack(anchor=E, pady=5)
    
    def update_image(self):
        """Snap a new image and update the display"""
        self.status_label.config(text="Capturing image...")
        
        try:
            image_array = snap_image()
            pil_image = process_image(image_array)
            
            if pil_image:
                self._display_image(pil_image)
                self.status_label.config(text="Image captured successfully")
                self.snap_button.config(text="Snap Again")
            else:
                self.status_label.config(text="Error: Unable to process image")
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
    
    def toggle_live_view(self):
        """Toggle the live view on/off"""
        if self.live_view_active:
            self.stop_event.set()
            if self.live_thread and self.live_thread.is_alive():
                self.live_thread.join(timeout=1.0)
            
            self.live_view_active = False
            self.progress.stop()
            self.live_button.config(text="Start Live View")
            self.snap_button.config(state="normal")
            self.save_button.config(state="normal")
            self.status_label.config(text="Live view stopped")
        else:
            self.live_view_active = True
            self.stop_event.clear()
            self.progress.start(10)
            self.live_button.config(text="Stop Live View")
            self.snap_button.config(state="disabled")
            self.save_button.config(state="disabled")
            self.status_label.config(text="Live view active")
            
            self.live_thread = threading.Thread(target=self.live_update_thread)
            self.live_thread.daemon = True
            self.live_thread.start()
    
    def live_update_thread(self):
        """Thread function for live view"""
        while not self.stop_event.is_set():
            try:
                image_array = snap_image()
                pil_image = process_image(image_array)
                
                if pil_image:
                    self.root.after(0, lambda img=pil_image: self._display_image(img))
                
                time.sleep(self.refresh_rate.get() / 1000.0)
            except Exception as e:
                self.root.after(0, lambda msg=str(e): self.status_label.config(text=f"Error: {msg}"))
                time.sleep(1)
    
    def _display_image(self, pil_image):
        """Update the image displayed on the canvas"""
        canvas_width = self.image_canvas.winfo_width() or 350
        canvas_height = self.image_canvas.winfo_height() or 250
        
        img_width, img_height = pil_image.size
        scale = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        resized_img = pil_image.resize((new_width, new_height), Image.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(resized_img)
        
        x_pos = (canvas_width - new_width) // 2
        y_pos = (canvas_height - new_height) // 2
        
        self.image_canvas.delete("all")
        self.image_canvas.create_image(x_pos, y_pos, anchor="nw", image=self.photo_image)
    
    def save_image(self):
        """Save the currently displayed image"""
        from tkinter import filedialog
        import datetime
        
        if not self.photo_image:
            self.status_label.config(text="No image to save")
            return
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"microscope_image_{timestamp}.png"
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_filename,
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            try:
                if hasattr(self, "original_pil_image") and self.original_pil_image:
                    self.original_pil_image.save(filepath)
                    self.status_label.config(text=f"Image saved to {filepath}")
                else:
                    self.status_label.config(text="No original image data to save")
            except Exception as e:
                self.status_label.config(text=f"Error saving image: {str(e)}")
    
    def _on_closing(self):
        """Handle window close event"""
        if self.live_view_active:
            self.toggle_live_view()
        self.root.destroy()
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    """Main function to start the application"""
    try:
        import ttkbootstrap
    except ImportError:
        import subprocess
        import sys
        print("The ttkbootstrap package is required. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ttkbootstrap"])
        import ttkbootstrap
    
    app = ModernMicroscopeViewer()
    app.run()

if __name__ == "__main__":
    main() 