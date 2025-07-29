import socket
import threading
import struct
from smart_wsi_scanner.smartpath import init_pycromanager
from smart_wsi_scanner.config import ConfigManager, sp_position
from smart_wsi_scanner.hardware import PycromanagerHardware
import sys
from smart_wsi_scanner.qp_server_config import Command, TCP_PORT, END_MARKER
import pathlib
import re
from smart_wsi_scanner.smartpath import smartpath
from smart_wsi_scanner.smartpath_qpscope import smartpath_qpscope
from pprint import pprint as dict_printer
import imageio
import numpy as np
import shutil


HOST = "0.0.0.0"  # Listen on all interfaces
PORT = TCP_PORT  # Arbitrary non-privileged port
shutdown_event = threading.Event()
wait_for_function_event = threading.Event()  # Event to wait for a function


def _pycromanager():
    """Initialize Pycro-Manager connection."""
    core, studio = init_pycromanager()
    if not core:
        print("Failed to initialize Micro-Manager connection")
        sys.exit(1)
    return core, studio


config_manager = ConfigManager()
ppm_settings = config_manager.get_config("config_PPM")
# TODO : Need stricter type checking for config

core, studio = _pycromanager()
hardware = PycromanagerHardware(core, ppm_settings, studio)
brushless = "KBD101_Thor_Rotation"


## Kinesis control for rotational stage for PPM
# TODO: need to move some of this to smartpath?
def ppm_to_thor(angle):
    return -2 * angle + 276


def thor_to_ppm(kinesis_pos):
    return (276 - kinesis_pos) / 2


def set_angle(theta):
    theta = ppm_to_thor(theta)
    core.set_position(brushless, theta)
    core.wait_for_device(brushless)


def read_tile_config(tile_config_path):
    positions = []
    if tile_config_path.exists():
        with open(tile_config_path, "r") as f:
            for line in f:
                ## only works with Gridstitcher format that Mike supplies from qupath extension
                pattern = r"^([\w\-\.]+); ; \(\s*([-\d.]+),\s*([-\d.]+)"
                # pattern = r"^([\w\-\.]+); ; \(\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)"
                m = re.match(pattern, line)
                if m:
                    z = None
                    filename = m.group(1)
                    x = float(m.group(2))
                    y = float(m.group(3))
                    try:
                        z = float(m.group(4))
                    except Exception as e:
                        print(e)
                        z = core.get_position()
                    # z = -23.0
                    positions.append([sp_position(x, y, z), filename])
    else:
        print(f"Tile configuration file {tile_config_path} does not exist.")
    return positions


def acquisitionWorkflow(message, test):

    parts = message.split(",")
    yaml_file_path = parts[0]
    projects_folder_path = parts[1]
    sample_label = parts[2]
    scan_type = parts[3]
    region_name = parts[4]
    angles_str = parts[5]
    modality = "_".join(scan_type.split("_")[:2])
    print(f"Angles arg: {angles_str}")
    # Remove parentheses and split by space
    ticks = [float(x) for x in angles_str.strip("()").split()]  ## TODO: cant use comma as delimiter
    print("  Modality:", modality)
    # print("Arguments received:")
    print("  YAML file:", yaml_file_path)
    print("  Projects folder:", projects_folder_path)
    print("  Sample label:", sample_label)
    print("  Scan type:", scan_type)
    print("  Region:", region_name)

    project_path = pathlib.Path(projects_folder_path) / sample_label
    output_path = project_path / scan_type / region_name
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    tile_config_path = output_path / "TileConfiguration.txt"
    positions = read_tile_config(tile_config_path)
    for tickname in ticks:
        pathlib.Path(output_path / str(tickname)).mkdir(exist_ok=True)
        shutil.copy2(tile_config_path, output_path / str(tickname) / "TileConfiguration.txt")

    sp = smartpath(core)
    print(f"Imaging {len(positions)} positions")
    for i, _p in enumerate(positions):
        pos, filename = _p
        hardware.move_to_position(pos)
        for tick in ticks:
            set_angle(tick)

            print(f"Set angle to {tick} ticks")
            image, metadata = hardware.snap_image()
            image_path = output_path / str(tick) / filename
            print(image_path)
            print(image_path.exists())
            if not image_path.parent.exists():
                image_path.parent.mkdir(parents=True, exist_ok=True)
            # This print statement is absolutely necessary for the progress bar in QuPath to work
            print("Tile saved: " + str(image_path), flush=True)
            # FIXME : change ddataclass to dict to read var modality
            if image_path.parent.exists():
                smartpath_qpscope.ome_writer(
                    filename=str(image_path),
                    pixel_size_um=ppm_settings.imagingMode.BF_10x.pixelSize_um,
                    data=image,
                )
                # np.save(image_path, image)
                # imageio.imwrite(str(image_path), image, format="tiff")
            else:
                print(f"Failed to save image at {image_path}. Directory does not exist.")

    current_props = sp.get_device_properties(core)
    with open(output_path / "MMproperties.txt", "w") as fid:
        dict_printer(current_props, stream=fid)
    # with open(output_path / "MM2_ImageTags_of_last_file.txt", "w") as fid:
    #    dict_printer(smartpath_qpscope.format_imagetags(metadata), stream=fid)
    print("Acquisition workflow completed.")
    wait_for_function_event.set()  # Signal that the acquisition workflow is done


def handle_client(conn, addr):
    print(f"Connected by client at {addr}")
    try:
        while True:
            data = conn.recv(8)
            print(data)
            if not data:
                break
            if data == Command.DISCONNECT.value:
                print(f"Client {addr} requested to quit.")
                break
            if data == Command.SHUTDOWN.value:
                print(f"Client {addr} requested server shutdown.")
                shutdown_event.set()
                break
            if data == Command.GETXY.value:
                current_position_xyz = hardware.get_current_position()
                response = struct.pack("!ff", current_position_xyz.x, current_position_xyz.y)
                conn.sendall(response)
                continue
            if data == Command.GETZ.value:
                current_position_xyz = hardware.get_current_position()
                response = struct.pack("!f", current_position_xyz.z)
                conn.sendall(response)
                continue
            if data == Command.GETR.value:
                kinesis_pos = core.get_position(brushless)
                response = struct.pack("!f", thor_to_ppm(kinesis_pos))
                conn.sendall(response)
                continue
            if data == Command.MOVER.value:
                coords = conn.recv(4)
                angle = struct.unpack("!f", coords)[0]
                print(f"Received from {addr}: {angle}")
                print(f"Client {addr} requested move to rotation angle: {angle}")
                new_angle = ppm_to_thor(angle)
                core.set_position(brushless, new_angle)
                core.wait_for_device(brushless)
                continue

            if data == Command.GET.value:
                message = ""
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    message += data.decode()
                    if str(END_MARKER) in message:
                        # Remove the escape string from the message
                        message = message.replace(END_MARKER, "")
                        break
                prop_name = message.split(",")[:-1]
                print(prop_name)
                response = struct.pack("!f", float(core.get_property(*prop_name)))
                conn.sendall(response)
                continue

            if data == Command.SET.value:
                # TODO
                print("will set prop later")
                continue

            if data == Command.MOVEZ.value:
                z = conn.recv(4)
                z_position = struct.unpack("!f", z)[0]
                print(f"Received from {addr}: {z_position}")
                print(f"Client {addr} requested move to Z position: {z_position}")
                hardware.move_to_position(sp_position(z=z_position))
                continue

            # Unpack float (network byte order)
            if data == Command.MOVE.value:
                coords = conn.recv(8)
                if len(coords) == 8:
                    x, y = struct.unpack("!ff", coords)
                    print(f"Received from {addr}: {(x,y)}")
                    print(f"Client {addr} requested move to: x={x}, y={y}")
                    hardware.move_to_position(sp_position(x, y))
                else:
                    print(f"Client {addr} sent incomplete move coordinates: {coords}")
                continue

            if data == Command.ACQUIRE.value:
                print(f"Client {addr} requested acquisition workflow.")
                message = ""
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    message += data.decode()
                    if str(END_MARKER) in message:
                        # Remove the escape string from the message
                        message = message.replace(END_MARKER, "")
                        break
                # print(len(message), message)
                function_thread = threading.Thread(
                    target=acquisitionWorkflow, args=(message, "test"), daemon=True
                )
                function_thread.start()
                # acquisitionWorkflow(message)
                wait_for_function_event.wait()
                continue

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
