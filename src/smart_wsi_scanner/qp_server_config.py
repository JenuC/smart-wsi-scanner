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
    GET = b"get_____"
    SET = b"set_____"
    ACQUIRE = b"acquire_"
    SHUTDOWN = b"shutdown"
    DISCONNECT = b"quitclnt"


for command in Command:
    if len(command.value) != 8:
        raise ValueError(f"Command {command.name} must be exactly 8 bytes long.")
    if not isinstance(command.value, bytes):
        raise TypeError(f"Command {command.name} must be of type bytes.")


# Extend the Command enum with new commands
class ExtendedCommand:
    """Extended commands for enhanced acquisition control."""

    # Existing commands from Command enum
    GETXY = Command.GETXY.value
    GETZ = Command.GETZ.value
    MOVEZ = Command.MOVEZ.value
    MOVE = Command.MOVE.value
    GETR = Command.GETR.value
    MOVER = Command.MOVER.value
    SHUTDOWN = Command.SHUTDOWN.value
    DISCONNECT = Command.DISCONNECT.value
    ACQUIRE = Command.ACQUIRE.value
    GET = Command.GET.value
    SET = Command.SET.value

    # New commands (8 bytes each)
    STATUS = b"status__"  # Get acquisition status
    PROGRESS = b"progress"  # Get acquisition progress
    CANCEL = b"cancel__"  # Cancel acquisition
