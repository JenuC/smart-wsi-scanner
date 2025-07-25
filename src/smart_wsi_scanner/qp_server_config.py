from enum import Enum


TCP_PORT = 5000  # Default port number for the server, can be changed as needed


class Command(Enum):
    GETXY = b"getxy___"
    GETZ = b"getz____"
    MOVE = b"move___"
    SHUTDOWN = b"shutdown"
    DISCONNECT = b"quitclnt"
