import java.io.*;
import java.net.*;
import java.nio.ByteBuffer;

public class StageControl {

    private static final String HOST = "127.0.0.1";  // Server address
    private static final int PORT = 5000;  // Server port

    public static void getStageXY() {
        try (Socket socket = new Socket(HOST, PORT);
             DataInputStream input = new DataInputStream(socket.getInputStream());
             DataOutputStream output = new DataOutputStream(socket.getOutputStream())) {

                    // Send raw bytes instead of using writeUTF
            byte[] message = "getxy___".getBytes(); // Convert the string to raw bytes
            output.write(message); // Send the raw bytes
            
            byte[] data = new byte[8];
            input.readFully(data);

            if (data.length == 8) {
                float x = ByteBuffer.wrap(data, 0, 4).getFloat();
                float y = ByteBuffer.wrap(data, 4, 4).getFloat();
                System.out.println("(" + x + ", " + y + ")");
            } else {
                System.out.println("Failed to receive stage location.");
            }

        } catch (IOException e) {
            System.out.println("Error in getStageXY: " + e.getMessage());
        }
    }

    public static void getStageZ() {
        try (Socket socket = new Socket(HOST, PORT);
             DataInputStream input = new DataInputStream(socket.getInputStream());
             DataOutputStream output = new DataOutputStream(socket.getOutputStream())) {

            output.writeUTF("getz____");
            byte[] data = new byte[4];
            input.readFully(data);

            if (data.length == 4) {
                float z = ByteBuffer.wrap(data).getFloat();
                System.out.println(z);
            } else {
                System.out.println("Failed to receive stage location.");
            }

        } catch (IOException e) {
            System.out.println("Error in getStageZ: " + e.getMessage());
        }
    }

    public static void moveStageXY(float x, float y) {
        try (Socket socket = new Socket(HOST, PORT);
             DataOutputStream output = new DataOutputStream(socket.getOutputStream())) {

            ByteBuffer buffer = ByteBuffer.allocate(8);
            buffer.putFloat(x);
            buffer.putFloat(y);
            output.write(buffer.array());

            System.out.println("Sending: " + x + ", " + y);

        } catch (IOException e) {
            System.out.println("Error in moveStageXY: " + e.getMessage());
        }
    }

    public static void shutdownServer() {
        try (Socket socket = new Socket(HOST, PORT);
             DataOutputStream output = new DataOutputStream(socket.getOutputStream())) {

            output.writeUTF("shutdown");
            System.out.println("Sent server shutdown command. Disconnected.");

        } catch (IOException e) {
            System.out.println("Error in shutdownServer: " + e.getMessage());
        }
    }

    public static void disconnect() {
        try (Socket socket = new Socket(HOST, PORT);
             DataOutputStream output = new DataOutputStream(socket.getOutputStream())) {

            output.writeUTF("quitclnt");
            System.out.println("Disconnected from server.");

        } catch (IOException e) {
            System.out.println("Error in disconnect: " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        // Example of how to use the methods

        // Move the stage to X=10, Y=20
        //moveStageXY(10.0f, 20.0f);

        // Get the current XY position
        getStageXY();

        // Get the current Z position
        //getStageZ();

        // Shutdown the server
        //shutdownServer();
        
        // Disconnect
        //disconnect();
    }
}
