import socket
import threading
import struct
from smart_wsi_scanner.smartpath import init_pycromanager
from smart_wsi_scanner.config import ConfigManager, sp_position
from smart_wsi_scanner.hardware import PycromanagerHardware
import argparse
import sys
import numpy as np
from skimage import img_as_ubyte, exposure

HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 5000  # Arbitrary non-privileged port
shutdown_event = threading.Event()

core, studio = init_pycromanager()
config_manager = ConfigManager()
if not core:
    print("Failed to initialize Micro-Manager connection")
ppm_settings = config_manager.get_config("config_PPM")
hardware = PycromanagerHardware(core, ppm_settings, studio)
brushless = "KBD101_Thor_Rotation"


def get_stageXY():
    current_position_xyz = hardware.get_current_position()
    print(f"{current_position_xyz.x,current_position_xyz.y}")


def get_stageZ():
    current_position_xyz = hardware.get_current_position()
    print(current_position_xyz.z)


def get_position():
    print(hardware.get_current_position())


def move_stageXY():
    parser = argparse.ArgumentParser(description="Move XYZ stage")

    # All arguments use flags and are not positional
    parser.add_argument("-x", "--x", type=float, required=True, help="X position")
    parser.add_argument("-y", "--y", type=float, required=True, help="Y position")
    parser.add_argument("-z", "--z", type=float, required=False, help="Z position (optional)")

    args = parser.parse_args()

    pos_kwargs = {"x": args.x, "y": args.y}
    if args.z is not None:
        pos_kwargs["z"] = args.z

    hardware.move_to_position(sp_position(**pos_kwargs))
    print(hardware.get_current_position())


def move_stageZ():
    parser = argparse.ArgumentParser(description="Move Z stage")
    parser.add_argument("-z", "--z", type=float, required=True, help="Z position")
    args = parser.parse_args()

    hardware.move_to_position(sp_position(z=args.z))
    print(hardware.get_current_position())


## Kinesis control for rotational stage for PPM
# TODO: need to move some of this to smartpath?
def ppm_to_thor(angle):
    return -2 * angle + 276


def thor_to_ppm(kinesis_pos):
    return (276 - kinesis_pos) / 2


def get_stageR():
    kinesis_pos = core.get_position(brushless)
    print(f"{thor_to_ppm(kinesis_pos):.2f}")


def move_stageR():
    """Move rotation stage to specified angle."""
    parser = argparse.ArgumentParser(description="Move rotation stage")
    parser.add_argument("angle", type=float, help="Rotation angle in degrees")
    args = parser.parse_args(sys.argv[2:])

    newAngle = ppm_to_thor(args.angle)
    core.set_position(brushless, newAngle)
    core.wait_for_device(brushless)
    get_stageR()


def acquisitionWorkflow():
    # TODO exec : minimal_qupathrunner_v3.py
    print("running acq from tile-config")
    # pass


def handle_client(conn, addr):
    print(f"Connected by {addr}")
    try:
        while True:
            data = conn.recv(8)
            print(data)
            if not data:
                break
            if data == b"quitclnt":
                print(f"Client {addr} requested to quit.")
                break
            if data == b"shutdown":
                print(f"Client {addr} requested server shutdown.")
                shutdown_event.set()
                break
            if data == b"getxy___":
                current_position_xyz = hardware.get_current_position()
                response = struct.pack("!ff", current_position_xyz.x, current_position_xyz.y)
                conn.sendall(response)
                continue
            if data == b"getz____":
                current_position_xyz = hardware.get_current_position()
                response = struct.pack("!f", current_position_xyz.z)
                conn.sendall(response)
                continue
            # Unpack float (network byte order)
            if data == b"move____":
                coords = conn.recv(8)
                if len(coords) == 8:
                    x, y = struct.unpack("!ff", coords)
                    print(f"Received from {addr}: {(x,y)}")
                    print(f"Client {addr} requested move to: x={x}, y={y}")
                    hardware.move_to_position(sp_position(x, y))
                else:
                    print(f"Client {addr} sent incomplete move coordinates: {coords}")
                continue
            # try:
            #    x, y = struct.unpack("!ff", data)
            #    print(f"Received from {addr}: {(x,y)}")
            #    hardware.move_to_position(sp_position(x, y))
            # except struct.error:
            #    print(f"Malformed data from {addr}: {data}")
    finally:
        conn.close()
        print(f"Connection closed for {addr}")


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")
        threads = []
        while not shutdown_event.is_set():
            try:
                s.settimeout(1.0)
                conn, addr = s.accept()
                thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
                thread.start()
                threads.append(thread)
            except socket.timeout:
                continue
            except OSError:
                break
        print("Server shutting down. Waiting for client threads to finish...")
        shutdown_event.set()
        for t in threads:
            t.join()
        print("Server has shut down.")


if __name__ == "__main__":
    main()
