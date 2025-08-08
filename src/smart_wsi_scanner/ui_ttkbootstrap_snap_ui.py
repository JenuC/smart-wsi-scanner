import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledFrame
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
        self.root.geometry("400x500")
        self.root.minsize(400, 500)
        
        # Variables
        self.photo_image = None
        self.live_view_active = False
        self.live_thread = None
        self.stop_event = threading.Event()
        self.refresh_rate = tk.IntVar(value=1000)
        
        # Section states
        self.settings_expanded = True
        self.joystick_expanded = True
        
        self._create_widgets()
        self._create_layout()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def stop_stage(self):
        """Stop the stage movement"""
        self.status_label.config(text="Stage stopped")
        # TODO: Implement actual stage stopping logic
    
    def _create_widgets(self):
        # Main container
        self.main_frame = ttk.Frame(self.root, padding=5)
        
        # Title
        self.title_label = ttk.Label(
            self.main_frame,
            text="Smart WSI Scanner",
            font=("Helvetica", 14, "bold")
        )
        
        # Image display
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_canvas = tk.Canvas(
            self.image_frame,
            width=350,
            height=200,
            bg="#1E1E1E",
            highlightthickness=1,
            highlightbackground="#0D6EFD"
        )
        
        # Status bar
        self.status_label = ttk.Label(
            self.main_frame,
            text="Ready",
            font=("Helvetica", 9)
        )
        
        # Control buttons
        self.button_frame = ttk.Frame(self.main_frame)
        self.snap_button = ttk.Button(
            self.button_frame,
            text="Snap Image",
            command=self.update_image,
            style="primary.TButton",
            width=12
        )
        self.live_button = ttk.Button(
            self.button_frame,
            text="Start Live View",
            command=self.toggle_live_view,
            width=12
        )
        self.save_button = ttk.Button(
            self.button_frame,
            text="Save Image",
            command=self.save_image,
            width=12
        )
        
        # Progress bar
        self.progress = ttk.Progressbar(
            self.main_frame,
            orient="horizontal",
            length=350,
            mode="indeterminate"
        )
        
        # Settings section
        self.settings_frame = ttk.LabelFrame(
            self.main_frame,
            text="Settings",
            padding=5
        )
        self.settings_toggle = ttk.Button(
            self.settings_frame,
            text="▼",
            width=3,
            command=lambda: self.toggle_section(self.settings_content, self.settings_toggle, "settings")
        )
        self.settings_content = ttk.Frame(self.settings_frame)
        
        # Refresh rate control
        self.refresh_label = ttk.Label(
            self.settings_content,
            text="Refresh Rate (ms):"
        )
        self.refresh_slider = ttk.Scale(
            self.settings_content,
            from_=100,
            to=2000,
            variable=self.refresh_rate,
            orient="horizontal",
            length=250
        )
        self.refresh_value = ttk.Label(
            self.settings_content,
            textvariable=self.refresh_rate
        )
        
        # Stage control section
        self.joystick_frame = ttk.LabelFrame(
            self.main_frame,
            text="Stage Control",
            padding=5
        )
        self.joystick_toggle = ttk.Button(
            self.joystick_frame,
            text="▼",
            width=3,
            command=lambda: self.toggle_section(self.joystick_content, self.joystick_toggle, "joystick")
        )
        self.joystick_content = ttk.Frame(self.joystick_frame)
        
        # Movement speed control
        self.movement_label = ttk.Label(
            self.joystick_content,
            text="Movement Speed"
        )
        self.movement_scale = ttk.Scale(
            self.joystick_content,
            from_=1,
            to=100,
            orient="horizontal",
            length=150,
            value=50
        )
        
        # Joystick buttons container
        self.joystick_container = ttk.Frame(self.joystick_content)
        
        # Joystick buttons
        self.joystick_buttons = {}
        directions = {
            'up': '↑',
            'down': '↓',
            'left': '←',
            'right': '→',
            'up_left': '↖',
            'up_right': '↗',
            'down_left': '↙',
            'down_right': '↘',
            'stop': '■'  # Add stop button
        }
        
        for direction, symbol in directions.items():
            if direction == 'stop':
                self.joystick_buttons[direction] = ttk.Button(
                    self.joystick_container,
                    text=symbol,
                    width=3,
                    style="danger.TButton",  # Red color for stop button
                    command=self.stop_stage
                )
            else:
                self.joystick_buttons[direction] = ttk.Button(
                    self.joystick_container,
                    text=symbol,
                    width=3,
                    command=lambda d=direction: self.move_stage(d)
                )
    
    def toggle_section(self, content_frame, toggle_button, section_name):
        """Toggle the visibility of a section's content with animation"""
        if section_name == "settings":
            self.settings_expanded = not self.settings_expanded
            is_expanded = self.settings_expanded
        else:
            self.joystick_expanded = not self.joystick_expanded
            is_expanded = self.joystick_expanded
        
        # Update toggle button text
        toggle_button.configure(text="▲" if is_expanded else "▼")
        
        if is_expanded:
            # Show content with animation
            content_frame.pack(fill=X, pady=5)
            self.animate_expand(content_frame)
        else:
            # Hide content with animation
            self.animate_collapse(content_frame)
    
    def animate_expand(self, frame):
        """Animate the expansion of a frame"""
        height = frame.winfo_reqheight()
        frame.configure(height=0)
        frame.pack_propagate(False)
        
        def update_height(current=0):
            if current < height:
                frame.configure(height=current)
                self.root.after(10, lambda: update_height(current + 5))
            else:
                frame.configure(height=height)
                frame.pack_propagate(True)
        
        update_height()
    
    def animate_collapse(self, frame):
        """Animate the collapse of a frame"""
        height = frame.winfo_height()
        frame.pack_propagate(False)
        
        def update_height(current=height):
            if current > 0:
                frame.configure(height=current)
                self.root.after(10, lambda: update_height(current - 5))
            else:
                frame.pack_forget()
                frame.pack_propagate(True)
        
        update_height()
    
    def _create_layout(self):
        # Main frame
        self.main_frame.pack(fill=BOTH, expand=YES)
        
        # Title
        self.title_label.pack(pady=(0, 5))
        
        # Image frame
        self.image_frame.pack(fill=BOTH, expand=YES, pady=5)
        self.image_canvas.pack(fill=BOTH, expand=YES)
        
        # Status
        self.status_label.pack(fill=X, pady=2)
        
        # Buttons
        self.button_frame.pack(fill=X, pady=5)
        self.snap_button.pack(side=LEFT, padx=2)
        self.live_button.pack(side=LEFT, padx=2)
        self.save_button.pack(side=LEFT, padx=2)
        
        # Progress bar
        self.progress.pack(fill=X, pady=2)
        
        # Settings layout
        self.settings_frame.pack(fill=X, pady=5)
        self.settings_toggle.pack(side=RIGHT, padx=5)
        self.settings_content.pack(fill=X, pady=5)
        self.refresh_label.pack(anchor=W)
        self.refresh_slider.pack(fill=X, pady=2)
        self.refresh_value.pack(anchor=E)

        # Stage control layout
        self.joystick_frame.pack(fill=X, pady=5)
        self.joystick_toggle.pack(side=RIGHT, padx=5)
        self.joystick_content.pack(fill=X, pady=5)
        self.movement_label.pack(anchor=W)
        self.movement_scale.pack(fill=X, pady=2)
        
        # Joystick container
        self.joystick_container.pack(pady=5)
        
        # Layout joystick buttons in a 3x3 grid
        button_positions = {
            'up_left': (0, 0), 'up': (0, 1), 'up_right': (0, 2),
            'left': (1, 0), 'stop': (1, 1), 'right': (1, 2),
            'down_left': (2, 0), 'down': (2, 1), 'down_right': (2, 2)
        }
        
        for direction, (row, col) in button_positions.items():
            self.joystick_buttons[direction].grid(row=row, column=col, padx=2, pady=2)
    
    def move_stage(self, direction):
        """Handle stage movement based on joystick direction"""
        speed = self.movement_scale.get()
        self.status_label.config(text=f"Moving stage {direction} at speed {speed}")
        # TODO: Implement actual stage movement logic
    
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
        canvas_height = self.image_canvas.winfo_height() or 200
        
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