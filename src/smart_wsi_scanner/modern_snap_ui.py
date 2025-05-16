import tkinter as tk
from tkinter import ttk
import threading
import time
from PIL import Image, ImageTk
import numpy as np
from skimage import img_as_ubyte, exposure
import sv_ttk  # Modern theme for tkinter

# Import the hardware functions from qp_snap
from smart_wsi_scanner.qp_snap import snap_image, process_image

class ModernMicroscopeViewer:
    def __init__(self, master=None):
        self.master = master or tk.Tk()
        self.master.title("Smart WSI Viewer")
        self.master.geometry("400x500")
        self.master.minsize(400, 500)
        
        # Apply modern SV-TTK theme
        sv_ttk.set_theme("dark")
        
        # Create style
        self.style = ttk.Style()
        
        # Configure colors
        self.bg_color = "#2E2E2E"
        self.accent_color = "#1E88E5"
        self.text_color = "#FFFFFF"
        
        self.master.configure(bg=self.bg_color)
        
        # Variables
        self.photo_image = None
        self.live_view_active = False
        self.live_thread = None
        self.stop_event = threading.Event()
        
        self._create_widgets()
        self._create_layout()
        
        # Handle window close
        self.master.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_widgets(self):
        # Create header frame
        self.header_frame = ttk.Frame(self.master)
        self.title_label = ttk.Label(
            self.header_frame, 
            text="Smart WSI Scanner", 
            font=("Segoe UI", 14, "bold"),
        )
        
        # Create image display frame
        self.image_frame = ttk.Frame(self.master)
        self.image_frame.config(width=350, height=250)
        
        # Create canvas for the image with border
        self.image_canvas = tk.Canvas(
            self.image_frame,
            width=350,
            height=250,
            bg="#1E1E1E",
            highlightthickness=1,
            highlightbackground=self.accent_color
        )
        
        # Status frame
        self.status_frame = ttk.Frame(self.master)
        self.status_label = ttk.Label(
            self.status_frame,
            text="Ready",
            font=("Segoe UI", 10)
        )
        
        # Controls frame
        self.controls_frame = ttk.Frame(self.master)
        
        # Progress bar for live view
        self.progress = ttk.Progressbar(
            self.controls_frame, 
            orient="horizontal", 
            length=350, 
            mode="indeterminate"
        )
        
        # Buttons
        self.snap_button = ttk.Button(
            self.controls_frame,
            text="Snap Image",
            command=self.update_image,
            style="Accent.TButton"
        )
        
        self.live_button = ttk.Button(
            self.controls_frame,
            text="Start Live View",
            command=self.toggle_live_view
        )
        
        self.save_button = ttk.Button(
            self.controls_frame,
            text="Save Image",
            command=self.save_image
        )
        
        # Settings frame
        self.settings_frame = ttk.LabelFrame(self.master, text="Settings")
        
        # Exposure slider
        self.exposure_label = ttk.Label(
            self.settings_frame,
            text="Refresh Rate (ms):"
        )
        
        self.refresh_rate = tk.IntVar(value=1000)
        self.refresh_slider = ttk.Scale(
            self.settings_frame,
            from_=100,
            to=2000,
            variable=self.refresh_rate,
            orient="horizontal",
            length=250
        )
        
        self.refresh_value_label = ttk.Label(
            self.settings_frame,
            textvariable=self.refresh_rate
        )
        
        # Footer frame
        self.footer_frame = ttk.Frame(self.master)
        self.version_label = ttk.Label(
            self.footer_frame, 
            text="v1.0.0", 
            font=("Segoe UI", 8)
        )
    
    def _create_layout(self):
        # Configure grid layout with padding
        self.master.columnconfigure(0, weight=1)
        padding = {"padx": 15, "pady": 5}
        
        # Header layout
        self.header_frame.grid(row=0, column=0, sticky="ew", **padding)
        self.header_frame.columnconfigure(0, weight=1)
        self.title_label.grid(row=0, column=0, pady=10)
        
        # Image frame layout
        self.image_frame.grid(row=1, column=0, sticky="nsew", **padding)
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.rowconfigure(0, weight=1)
        self.image_canvas.grid(row=0, column=0, sticky="nsew")
        
        # Status layout
        self.status_frame.grid(row=2, column=0, sticky="ew", **padding)
        self.status_frame.columnconfigure(0, weight=1)
        self.status_label.grid(row=0, column=0, sticky="w")
        
        # Controls layout
        self.controls_frame.grid(row=3, column=0, sticky="ew", **padding)
        self.controls_frame.columnconfigure(0, weight=1)
        self.controls_frame.columnconfigure(1, weight=1)
        self.controls_frame.columnconfigure(2, weight=1)
        
        self.snap_button.grid(row=0, column=0, sticky="ew", padx=5)
        self.live_button.grid(row=0, column=1, sticky="ew", padx=5)
        self.save_button.grid(row=0, column=2, sticky="ew", padx=5)
        self.progress.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        
        # Settings layout
        self.settings_frame.grid(row=4, column=0, sticky="ew", **padding)
        self.settings_frame.columnconfigure(1, weight=1)
        
        self.exposure_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.refresh_slider.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.refresh_value_label.grid(row=0, column=2, sticky="e", padx=5, pady=5)
        
        # Footer layout
        self.footer_frame.grid(row=5, column=0, sticky="ew", **padding)
        self.footer_frame.columnconfigure(0, weight=1)
        self.version_label.grid(row=0, column=0, sticky="e")
        
    def update_image(self):
        """Snap a new image and update the display"""
        self.status_label.config(text="Capturing image...")
        
        try:
            # Snap new image
            image_array = snap_image()
            
            # Process image
            pil_image = process_image(image_array)
            
            if pil_image:
                # Update the displayed image
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
            # Stop live view
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
            # Start live view
            self.live_view_active = True
            self.stop_event.clear()
            self.progress.start(10)
            self.live_button.config(text="Stop Live View")
            self.snap_button.config(state="disabled")
            self.save_button.config(state="disabled")
            self.status_label.config(text="Live view active")
            
            # Start live view in a separate thread
            self.live_thread = threading.Thread(target=self.live_update_thread)
            self.live_thread.daemon = True
            self.live_thread.start()
    
    def live_update_thread(self):
        """Thread function for live view"""
        while not self.stop_event.is_set():
            try:
                # Snap new image
                image_array = snap_image()
                
                # Process image
                pil_image = process_image(image_array)
                
                if pil_image:
                    # Update the displayed image
                    self.master.after(0, lambda img=pil_image: self._display_image(img))
                
                # Wait for the specified refresh rate
                time.sleep(self.refresh_rate.get() / 1000.0)
            except Exception as e:
                # Update the status label on error
                self.master.after(0, lambda msg=str(e): self.status_label.config(text=f"Error: {msg}"))
                time.sleep(1)  # Brief pause on error
    
    def _display_image(self, pil_image):
        """Update the image displayed on the canvas"""
        # Resize the image to fit the canvas if needed
        canvas_width = self.image_canvas.winfo_width() or 350
        canvas_height = self.image_canvas.winfo_height() or 250
        
        # Scale the image to fit the canvas while maintaining aspect ratio
        img_width, img_height = pil_image.size
        scale = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize the image
        resized_img = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        self.photo_image = ImageTk.PhotoImage(resized_img)
        
        # Calculate position to center the image
        x_pos = (canvas_width - new_width) // 2
        y_pos = (canvas_height - new_height) // 2
        
        # Clear previous image and display the new one
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
                # Get the original PIL image from the canvas
                # This is a simplified approach - in a real app you might
                # want to save the original unmodified image
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
            self.toggle_live_view()  # Stop live view if active
        self.master.destroy()
    
    def run(self):
        """Run the application"""
        self.master.mainloop()


def main():
    """Main function to start the application"""
    try:
        # Import at runtime to handle absence
        import sv_ttk
    except ImportError:
        import subprocess
        import sys
        print("The sv_ttk package is required for the modern UI. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sv-ttk"])
        import sv_ttk
    
    # Start the app
    app = ModernMicroscopeViewer()
    app.run()

if __name__ == "__main__":
    main() 