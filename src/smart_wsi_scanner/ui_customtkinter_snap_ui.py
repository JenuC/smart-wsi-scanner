import tkinter as tk
import customtkinter as ctk
import threading
import time
from PIL import Image, ImageTk
import numpy as np
from skimage import img_as_ubyte, exposure

# Import the hardware functions from qp_snap
from smart_wsi_scanner.qp_snap import snap_image, process_image

class ModernMicroscopeViewer:
    def __init__(self):
        # Set appearance mode and default color theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.root = ctk.CTk()
        self.root.title("Smart WSI Scanner")
        self.root.geometry("500x700")
        self.root.minsize(500, 700)
        
        # Variables
        self.photo_image = None
        self.live_view_active = False
        self.live_thread = None
        self.stop_event = threading.Event()
        self.refresh_rate = ctk.IntVar(value=1000)
        
        self._create_widgets()
        self._create_layout()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_widgets(self):
        # Main container
        self.main_frame = ctk.CTkFrame(self.root)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Smart WSI Scanner",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        
        # Image display
        self.image_frame = ctk.CTkFrame(self.main_frame)
        self.image_canvas = tk.Canvas(
            self.image_frame,
            width=450,
            height=300,
            bg="#2B2B2B",
            highlightthickness=0
        )
        
        # Status bar
        self.status_label = ctk.CTkLabel(
            self.main_frame,
            text="Ready",
            font=ctk.CTkFont(size=12)
        )
        
        # Control buttons
        self.button_frame = ctk.CTkFrame(self.main_frame)
        self.snap_button = ctk.CTkButton(
            self.button_frame,
            text="Snap Image",
            command=self.update_image,
            width=140,
            height=32
        )
        self.live_button = ctk.CTkButton(
            self.button_frame,
            text="Start Live View",
            command=self.toggle_live_view,
            width=140,
            height=32
        )
        self.save_button = ctk.CTkButton(
            self.button_frame,
            text="Save Image",
            command=self.save_image,
            width=140,
            height=32
        )
        
        # Progress bar
        self.progress = ctk.CTkProgressBar(
            self.main_frame,
            width=450,
            mode="indeterminate"
        )
        self.progress.set(0)
        
        # Settings frame
        self.settings_frame = ctk.CTkFrame(self.main_frame)
        self.settings_label = ctk.CTkLabel(
            self.settings_frame,
            text="Settings",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        
        # Refresh rate control
        self.refresh_label = ctk.CTkLabel(
            self.settings_frame,
            text="Refresh Rate (ms):",
            font=ctk.CTkFont(size=12)
        )
        self.refresh_slider = ctk.CTkSlider(
            self.settings_frame,
            from_=100,
            to=2000,
            number_of_steps=19,
            variable=self.refresh_rate,
            width=300
        )
        self.refresh_value = ctk.CTkLabel(
            self.settings_frame,
            textvariable=self.refresh_rate,
            font=ctk.CTkFont(size=12)
        )
        
        # Theme switcher
        self.theme_label = ctk.CTkLabel(
            self.settings_frame,
            text="Theme:",
            font=ctk.CTkFont(size=12)
        )
        self.theme_switch = ctk.CTkSwitch(
            self.settings_frame,
            text="Light Mode",
            command=self._toggle_theme,
            font=ctk.CTkFont(size=12)
        )
        
        # Version label
        self.version_label = ctk.CTkLabel(
            self.main_frame,
            text="v1.0.0",
            font=ctk.CTkFont(size=10)
        )
    
    def _create_layout(self):
        # Main frame
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        self.title_label.pack(pady=(0, 20))
        
        # Image frame
        self.image_frame.pack(fill="both", expand=True, pady=10)
        self.image_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Status
        self.status_label.pack(pady=10)
        
        # Buttons
        self.button_frame.pack(fill="x", pady=10)
        self.snap_button.pack(side="left", padx=5, expand=True)
        self.live_button.pack(side="left", padx=5, expand=True)
        self.save_button.pack(side="left", padx=5, expand=True)
        
        # Progress bar
        self.progress.pack(pady=10)
        
        # Settings
        self.settings_frame.pack(fill="x", pady=10, padx=10)
        self.settings_label.pack(pady=(10, 20))
        
        # Refresh rate controls
        self.refresh_label.pack(anchor="w", padx=20)
        self.refresh_slider.pack(fill="x", padx=20, pady=5)
        self.refresh_value.pack(anchor="e", padx=20)
        
        # Theme controls
        self.theme_label.pack(anchor="w", padx=20, pady=(20, 5))
        self.theme_switch.pack(anchor="w", padx=20)
        
        # Version
        self.version_label.pack(anchor="e", pady=10)
    
    def _toggle_theme(self):
        """Toggle between light and dark theme"""
        if ctk.get_appearance_mode() == "dark":
            ctk.set_appearance_mode("light")
            self.theme_switch.configure(text="Dark Mode")
        else:
            ctk.set_appearance_mode("dark")
            self.theme_switch.configure(text="Light Mode")
    
    def update_image(self):
        """Snap a new image and update the display"""
        self.status_label.configure(text="Capturing image...")
        
        try:
            image_array = snap_image()
            pil_image = process_image(image_array)
            
            if pil_image:
                self._display_image(pil_image)
                self.status_label.configure(text="Image captured successfully")
                self.snap_button.configure(text="Snap Again")
            else:
                self.status_label.configure(text="Error: Unable to process image")
        except Exception as e:
            self.status_label.configure(text=f"Error: {str(e)}")
    
    def toggle_live_view(self):
        """Toggle the live view on/off"""
        if self.live_view_active:
            self.stop_event.set()
            if self.live_thread and self.live_thread.is_alive():
                self.live_thread.join(timeout=1.0)
            
            self.live_view_active = False
            self.progress.stop()
            self.live_button.configure(text="Start Live View")
            self.snap_button.configure(state="normal")
            self.save_button.configure(state="normal")
            self.status_label.configure(text="Live view stopped")
        else:
            self.live_view_active = True
            self.stop_event.clear()
            self.progress.start()
            self.live_button.configure(text="Stop Live View")
            self.snap_button.configure(state="disabled")
            self.save_button.configure(state="disabled")
            self.status_label.configure(text="Live view active")
            
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
                self.root.after(0, lambda msg=str(e): self.status_label.configure(text=f"Error: {msg}"))
                time.sleep(1)
    
    def _display_image(self, pil_image):
        """Update the image displayed on the canvas"""
        canvas_width = self.image_canvas.winfo_width() or 450
        canvas_height = self.image_canvas.winfo_height() or 300
        
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
            self.status_label.configure(text="No image to save")
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
                    self.status_label.configure(text=f"Image saved to {filepath}")
                else:
                    self.status_label.configure(text="No original image data to save")
            except Exception as e:
                self.status_label.configure(text=f"Error saving image: {str(e)}")
    
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
        import customtkinter
    except ImportError:
        import subprocess
        import sys
        print("The customtkinter package is required. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "customtkinter"])
        import customtkinter
    
    app = ModernMicroscopeViewer()
    app.run()

if __name__ == "__main__":
    main() 