from enum import Enum


TCP_PORT = 5000  # Default port number for the server, can be changed as needed
END_MARKER = "ENDOFSTR"


class Command(Enum):
    GETXY = b"getxy___"
    GETZ = b"getz____"
    MOVEZ = b"move_z__"
    GETR = b"getr____"
    MOVER = b"move_r__"
    MOVE = b"move____"
    ACQUIRE = b"acquire_"
    SHUTDOWN = b"shutdown"
    DISCONNECT = b"quitclnt"


for command in Command:
    if len(command.value) != 8:
        raise ValueError(f"Command {command.name} must be exactly 8 bytes long.")
    if not isinstance(command.value, bytes):
        raise TypeError(f"Command {command.name} must be of type bytes.")
