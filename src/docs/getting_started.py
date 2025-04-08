# %% [markdown]
# # Getting Started with Smart WSI Scanner
#
# This guide demonstrates how to use the Smart WSI Scanner package for automated whole slide imaging.
# The package provides a high-level interface for controlling microscopes using Pycromanager.

# %% [markdown]
# ## Basic Setup
#
# First, let's import the necessary modules and initialize the hardware:

# %%
from pycromanager import Core
from smart_wsi_scanner import PycromanagerHardware, ConfigManager
from smart_wsi_scanner.config import sp_position

# Initialize Pycromanager core
core = Core()

# Create configuration manager and load settings
config_manager = ConfigManager()
camm_config = config_manager.get_config('config_CAMM')

# Initialize hardware with configuration
hardware = PycromanagerHardware(core, camm_config)

# %% [markdown]
# ## Stage Control
#
# Let's try moving the microscope stage to a specific position:

# %%
# Create a position object
position = sp_position(x=100, y=100, z=0)

# Move to the position
hardware.move_to_position(position)

# Get current position
current_pos = hardware.get_current_position()
print(f"Current position: x={current_pos.x}, y={current_pos.y}, z={current_pos.z}")

# %% [markdown]
# ## Image Capture
#
# Now let's capture an image and examine its metadata:

# %%
import matplotlib.pyplot as plt

# Capture image
image, metadata = hardware.snap_image()

# Display image
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.title("Captured Image")
plt.axis('off')
plt.show()

# Print metadata
print("\nImage Metadata:")
for key, value in metadata.items():
    print(f"{key}: {value}")

# %% [markdown]
# ## Objective Control
#
# Let's try changing the objective lens:

# %%
# Change to 4X objective
hardware.set_objective("4X")

# %% [markdown]
# ## Autofocus
#
# Finally, let's perform an autofocus operation:

# %%
# Perform autofocus with default parameters
best_focus = hardware.autofocus()
print(f"Best focus position: {best_focus}")

# %% [markdown]
# ## Advanced Usage
#
# For more advanced usage, including:
# - Custom autofocus parameters
# - White balance correction
# - Background correction
# - Multi-position imaging
#
# Please refer to the examples in the `src/examples` directory. 