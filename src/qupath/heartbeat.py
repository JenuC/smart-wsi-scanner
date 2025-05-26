import socket
import argparse
import time
import sys

# linked to https://github.com/MichaelSNelson/qupath-extension-qpsc/commit/58ebff3b26b697aca1813d13033a22c05583441d
# qupath .java test-ectension will call this heartbeat with a port that qupath will listen
# python process is spawn out and sending heartbeat to a fixed port opened by qupath:
# if port is not there python code breaks
# if python is not sending heartbeat, it detects that.
# if port is open, python will keep on sending heartbeats with sleep(2)
# no qupath port
# Python: Could not connect to QuPath: [WinError 10061] No connection could be made because the target machine actively refused it                            
# sending to port
# Python: Connected to QuPath workflow, sending heartbeats.                                                                                                   
# Python: Lost connection to QuPath workflow, exiting.

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--host", type=str, required=True)
    # parser.add_argument("--port", type=int, required=True)
    # args = parser.parse_args()
    # host,port = args.host, args.port
    host ="127.0.0.1" 
    port = 53717
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host,port))
        print("Python: Connected to QuPath workflow, sending heartbeats.")
        while True:
            try:
                s.sendall(b"heartbeat\n")
                time.sleep(2)
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                print("Python: Lost connection to QuPath workflow, exiting.")
                break
    except Exception as ex:
        print(f"Python: Could not connect to QuPath: {ex}")
        sys.exit(1)
    finally:
        s.close()

if __name__ == "__main__":
    main()
