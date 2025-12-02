import socket

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the host and port to listen on
host = '127.0.0.1'
port = 12345

# Bind the socket to the host and port
server_socket.bind((host, port))

# Listen for incoming connections
server_socket.listen(1)

# Accept a connection from a client
client_socket, client_address = server_socket.accept()
print('Connected by', client_address)

# Receive and process data from the client
while True:
    data = client_socket.recv(1024)
    if not data:
        break
    # Process the received data
    print('Received:', data.decode())

# Close the connection
client_socket.close()
server_socket.close()
