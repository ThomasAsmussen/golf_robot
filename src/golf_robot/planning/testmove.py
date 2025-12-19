import socket
import time
from kinematics import move_point_xyz
import numpy as np

print(f"Start : {time.ctime()}")

HOST1 = '192.38.66.227'    # UR10
# HOST1 = '10.0.2.15' 
# HOST1 = "127.0.0.1"
PORT1 = 30003              # The same port as used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST1, PORT1))
# s.send(b'movej([1,-1.4 ,1,-1.57, 1,0],0.5,0.5)\n')
# s.send(b'movej([0, -0.3581, -0.6049, -2.1786, 0, 0],0.2,0.2)\n')
# s.send(b'movej([-2.7813, -2.7835, 0.6049, -0.9630, 2.7813, 0],0.5,0.5)\n')
q_start = np.array([np.deg2rad(-126.33), np.deg2rad(-139.79), np.deg2rad(-96.62), np.deg2rad(55.14), np.deg2rad(35.87), np.deg2rad(20.47)], dtype=float)

# q_start = np.array([-2.18539977, -2.44831034, -1.85033569,  1.15618144,  0.61460362,  0.50125766], dtype=float)
q_start = move_point_xyz(0.05, -0.057, -0.005, q_start, q_start)[0]
print(q_start)
q_start = move_point_xyz(-0.2, -0.2, 0.0, q_start, q_start)[0]

s.send(f'movej({q_start.tolist()},0.2,0.2)\n'.encode())
# s.send(b'movej([-2.18539977, -2.44831034, -1.85033569,  1.15618144,  0.61460362,  0.50125766],0.5,0.5)\n')
# [-2.18539977 -2.40191466 -1.86085842  1.1203085   0.61460362  0.50125766]
time.sleep(5)

s.close()
print(f"End : {time.ctime()}")
#print 'Received', repr(data)
