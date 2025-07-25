import socket
import struct
import argparse
import sys
from smart_wsi_scanner.qp_server_config import Command, TCP_PORT, END_MARKER

HOST = "127.0.0.1"  # Server address (localhost by default)
PORT = TCP_PORT  # Must match server


def get_stageXY():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(Command.GETXY.value)
        data = s.recv(8)
        if len(data) == 8:
            x, y = struct.unpack("!ff", data)
            print(f"{x,y}")
        else:
            print("Failed to receive stage location.")


def get_stageZ():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(Command.GETZ.value)
        data = s.recv(4)
        if len(data) == 4:
            z = struct.unpack("!f", data)
            print(f"{z}")
        else:
            print("Failed to receive stage location.")


def move_stageZ():
    parser = argparse.ArgumentParser(description="Move Z stage")
    parser.add_argument("-z", "--z", type=float, required=True, help="Z position in microns")
    args = parser.parse_args()
    packed = struct.pack("!f", args.z)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(Command.MOVEZ.value + packed)


def move_stageXY():

    parser = argparse.ArgumentParser(description="Move XYZ stage")

    # All arguments use flags and are not positional
    parser.add_argument("-x", "--x", type=float, required=True, help="X position")
    parser.add_argument("-y", "--y", type=float, required=True, help="Y position")

    args = parser.parse_args()

    x, y = args.x, args.y
    packed = struct.pack("!ff", x, y)
    # print("Asking to move to", x, y)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(Command.MOVE.value + packed)


def get_stageR():
    """Get the current rotation angle of the stage."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(Command.GETR.value)
        data = s.recv(4)
        if len(data) == 4:
            angle = struct.unpack("!f", data)[0]
            print(f"Current rotation angle: {angle:.2f} degrees")
        else:
            print("Failed to receive rotation angle.")


def move_stageR():
    """Move rotation stage to specified angle."""
    parser = argparse.ArgumentParser(description="Move rotation stage")
    parser.add_argument("angle", type=float, help="Rotation angle in degrees")
    args = parser.parse_args(sys.argv[2:])

    packed = struct.pack("!f", args.angle)
    # print("Asking to move to", x, y)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(Command.MOVER.value + packed)


def shutdown_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(Command.SHUTDOWN.value)
        print("Sent server shutdown command. Disconnected.")


def disconnect():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(Command.DISCONNECT.value)
        print("Disconnected from server.")


def acquisition_workflow():
    """Run the acquisition workflow."""

    if len(sys.argv) > 5:
        (
            _,
            yaml_file_path,
            projects_folder_path,
            sample_label,
            scan_type,
            region_name,
            angles_str,
        ) = sys.argv
    else:
        print(
            "Usage: acq workflow needs following: <yaml_file_path> <projects_folder_path> <sample_label> <scan_type> <region_name> [angles_str]"
        )
        return
    data = [yaml_file_path, projects_folder_path, sample_label, scan_type, region_name, angles_str]
    message = ",".join(data) + "," + END_MARKER
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(Command.ACQUIRE.value + message.encode())
        print("Acquisition workflow started.")


def main():
    while True:
        user_input = input("Enter Q (quit) or D(disconnect) : ")
        if user_input == "Q":
            shutdown_server()
            break
        elif user_input == "D":
            disconnect()
            break
        elif user_input == "XY":
            get_stageXY()
            continue
        elif user_input == "Z":
            get_stageZ()
            continue
        elif user_input == "R":
            get_stageR()
            continue
        else:
            print("Invalid commands. Please try again.")


if __name__ == "__main__":

    # Uncomment the following lines to test individual functions
    # move_stageR()
    # move_stageZ()
    # move_stageXY()
    # acquisitionWorkflow()

    ## all others are available via command line args
    main()
