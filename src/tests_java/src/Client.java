import java.io.*;
import java.net.*;
import java.nio.ByteBuffer;
import java.util.Scanner;

public class Client {

    // Server details
    private static final String HOST = "127.0.0.1";  // Server address
    private static final int PORT = 5000;            // Port number (same as in Python example)
    
    // Enum for Commands (simulating the Command class in Python)
    public enum Command {
        GETXY("getxy___"),
        GETZ("getz____"),
        MOVEZ("movez___"),
        MOVE("move___"),
        GETR("getr____"),
        MOVER("mover__"),
        SHUTDOWN("shutdown"),
        DISCONNECT("quitclnt"),
        ACQUIRE("acquire__");

        private final String value;

        Command(String value) {
            this.value = value;
        }

        public String getValue() {
            return value;
        }
    }

    // Get the XY position of the stage
    public static void getStageXY() {
        try (Socket socket = new Socket(HOST, PORT);
             DataInputStream input = new DataInputStream(socket.getInputStream());
             DataOutputStream output = new DataOutputStream(socket.getOutputStream())) {

            output.write(Command.GETXY.getValue().getBytes());
            byte[] data = new byte[8];  // Receive 8 bytes for 2 floats
            input.readFully(data);

            if (data.length == 8) {
                float x = ByteBuffer.wrap(data, 0, 4).getFloat();
                float y = ByteBuffer.wrap(data, 4, 4).getFloat();
                System.out.println("Stage XY position: (" + x + ", " + y + ")");
            } else {
                System.out.println("Failed to receive stage location.");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Get the Z position of the stage
    public static void getStageZ() {
        try (Socket socket = new Socket(HOST, PORT);
             DataInputStream input = new DataInputStream(socket.getInputStream());
             DataOutputStream output = new DataOutputStream(socket.getOutputStream())) {

            output.write(Command.GETZ.getValue().getBytes());
            byte[] data = new byte[4];  // Receive 4 bytes for 1 float
            input.readFully(data);

            if (data.length == 4) {
                float z = ByteBuffer.wrap(data).getFloat();
                System.out.println("Stage Z position: " + z);
            } else {
                System.out.println("Failed to receive stage location.");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Move the Z stage to a specific position
    public static void moveStageZ(float z) {
        try (Socket socket = new Socket(HOST, PORT);
             DataOutputStream output = new DataOutputStream(socket.getOutputStream())) {

            byte[] packed = ByteBuffer.allocate(4).putFloat(z).array();
            output.write(Command.MOVEZ.getValue().getBytes());
            output.write(packed);  // Send the packed data
            System.out.println("Moving stage Z to: " + z);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Move the XY stage to specific positions
    public static void moveStageXY(float x, float y) {
        try (Socket socket = new Socket(HOST, PORT);
             DataOutputStream output = new DataOutputStream(socket.getOutputStream())) {

            byte[] packed = ByteBuffer.allocate(8).putFloat(x).putFloat(y).array();
            output.write(Command.MOVE.getValue().getBytes());
            output.write(packed);  // Send the packed data
            System.out.println("Moving stage XY to: (" + x + ", " + y + ")");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Get the current rotation angle
    public static void getStageR() {
        try (Socket socket = new Socket(HOST, PORT);
             DataInputStream input = new DataInputStream(socket.getInputStream());
             DataOutputStream output = new DataOutputStream(socket.getOutputStream())) {

            output.write(Command.GETR.getValue().getBytes());
            byte[] data = new byte[4];  // Receive 4 bytes for 1 float
            input.readFully(data);

            if (data.length == 4) {
                float angle = ByteBuffer.wrap(data).getFloat();
                System.out.println("Current rotation angle: " + angle + " degrees");
            } else {
                System.out.println("Failed to receive rotation angle.");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Move the rotation stage to a specific angle
    public static void moveStageR(float angle) {
        try (Socket socket = new Socket(HOST, PORT);
             DataOutputStream output = new DataOutputStream(socket.getOutputStream())) {

            byte[] packed = ByteBuffer.allocate(4).putFloat(angle).array();
            output.write(Command.MOVER.getValue().getBytes());
            output.write(packed);  // Send the packed data
            System.out.println("Moving rotation stage to: " + angle + " degrees");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Shutdown the server
    public static void shutdownServer() {
        try (Socket socket = new Socket(HOST, PORT);
             DataOutputStream output = new DataOutputStream(socket.getOutputStream())) {

            output.write(Command.SHUTDOWN.getValue().getBytes());
            System.out.println("Sent shutdown command to server.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Disconnect from the server
    public static void disconnect() {
        try (Socket socket = new Socket(HOST, PORT);
             DataOutputStream output = new DataOutputStream(socket.getOutputStream())) {

            output.write(Command.DISCONNECT.getValue().getBytes());
            System.out.println("Disconnected from server.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Main method handling user input
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        while (true) {
            System.out.print("Enter Q (quit), D (disconnect), XY, Z, R: ");
            String userInput = scanner.nextLine().toUpperCase();

            if (userInput.equals("Q")) {
                shutdownServer();
                break;
            } else if (userInput.equals("D")) {
                disconnect();
                break;
            } else if (userInput.equals("XY")) {
                getStageXY();
            } else if (userInput.equals("Z")) {
                getStageZ();
            } else if (userInput.equals("R")) {
                getStageR();
            } else {
                System.out.println("Invalid command. Please try again.");
            }
        }
        scanner.close();
    }
}
