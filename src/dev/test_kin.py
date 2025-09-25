# from pylablib.devices import Thorlabs

# print(Thorlabs.list_kinesis_devices())


# stage = Thorlabs.KinesisMotor("28253598")
# #print(stage.get_velocity_parameters) #doesn't return readable info
# print(stage.get_position())
# stage.move_to(360)
# stage.wait_for_stop()
# print(stage.get_position())
# stage.close()
###

from pylablib.devices import Thorlabs

# # List devices to confirm connection
# print("Available devices:", Thorlabs.list_kinesis_devices())

# # Connect to the stage
# stage = Thorlabs.KinesisMotor("28253598")

# try:
#     # Check if stage is homed
#     if not stage.is_homed():
#         print("Homing stage...")
#         stage.home(sync=True)  # Wait for homing to complete
#         print("Homing complete")
    
#     # Get current position
#     print(f"Current position: {stage.get_position()}")
    
#     # Check velocity parameters
#     vel_params = stage.get_velocity_parameters()
#     print(f"Velocity parameters: {vel_params}")
    
#     # Move to position
#     print("Moving to position 360...")
#     stage.move_to(00)
#     stage.wait_for_stop()
    
#     # Confirm new position
#     print(f"New position: {stage.get_position()}")
    
# finally:
#     stage.close()
    
    
# More detailed example with debugging
stage = Thorlabs.KinesisMotor("28253598", scale="DDR25")
print(f"Detected stage: {stage.get_stage()}")
print(f"Scale units: {stage.get_scale_units()}")
print(f"Scale factors: {stage.get_scale()}")
try:
    # Check current enable state
    print(f"Status: {stage.get_status()}")
    
    # Try to enable the channel (even if it shows as enabled)
    print("\nEnabling motor...")
    stage._enable_channel(True)  
    
    # Wait a moment for it to take effect
    import time
    time.sleep(1)
    
    # Check status again
    print(f"Status after enable: {stage.get_status()}")
    
    # Now try a simple movement
    current = stage.get_position()
    print(f"\nCurrent position: {current}")
    
    # Move by a small amount
    target = current + 10
    print(f"Moving to {target}...")
    
    stage.move_to(target)
    
    # Wait with status monitoring
    for i in range(20):  # 10 seconds max
        if not stage.is_moving():
            print("Movement complete")
            break
        status = stage.get_status()
        pos = stage.get_position()
        print(f"Moving... pos={pos}, status={status}")
        time.sleep(0.5)
    
    print(f"Final position: {stage.get_position()}")
    
finally:
    stage.close()