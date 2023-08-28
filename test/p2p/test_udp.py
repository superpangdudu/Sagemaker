
import socket

# write a UDP client to post a stun request to a server
def stun_request(host, port):
    # create a socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # bind port
    sock.bind(('192.168.0.106', 12345))
    # send message to server
    sock.sendto('hello', ('47.104.154.71', 12600))
    # receive message from server
    data, addr = sock.recvfrom(1024)
    # print server IP and port
    print(data)
    print(addr)

