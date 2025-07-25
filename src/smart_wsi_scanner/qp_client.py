import socket
import struct
import argparse
import sys
from smart_wsi_scanner.qp_server_config import Command, TCP_PORT

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


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print(f"Connected to server at {HOST}:{PORT}")
        while True:
            user_input = input("Enter: ")
            try:
                value = float(user_input)
                packed = struct.pack("!f", value)
                s.sendall(packed)
            except ValueError:
                print("Invalid input. Please enter a valid float, 'quit', or 'close'.")


if __name__ == "__main__":
    # main()
    # get_stage_xy()
    shutdown_server()

    # move_stageXY()
    # get_stageXY()
