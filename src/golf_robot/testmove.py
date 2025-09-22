import socket
import time

print(f"Start : {time.ctime()}")

# HOST1 = '192.38.66.227'    # UR10
# HOST1 = '10.0.2.15' 
HOST1 = "127.0.0.1"
PORT1 = 30003              # The same port as used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST1, PORT1))
# s.send(b'movej([1,-1.4 ,1,-1.57, 1,0],0.5,0.5)\n')
# s.send(b'movej([0, -0.3581, -0.6049, -2.1786, 0, 0],0.2,0.2)\n')
s.send(b'movej([-2.7813, -2.7835, 0.6049, -0.9630, 2.7813, 0],0.5,0.5)\n')
time.sleep(5)

s.close()
print(f"End : {time.ctime()}")
#print 'Received', repr(data)
