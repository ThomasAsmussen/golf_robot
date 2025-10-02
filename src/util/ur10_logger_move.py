import socket, time
from ur10_logger import UR10Logger

HOST = "192.168.56.101"
#HOST = '192.38.66.227'    # UR10

# 1) start telemetry on 30003
ur = UR10Logger(HOST, port=30003)
ur.connect()
ur.start_logging()  # non-blocking

# 2) send program on 30002
cmd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
cmd.connect((HOST, 30002))
roll, pitch, yaw = 0.8161, -1.4130, -0.8167
prog = (
    "def swing():\n"
    "  movej([-2.095,-2.618,-1.745,1.22,0.5236,-2.618],0.2,0.2)\n"
    f"  movel(p[-0.997,-0.546,-0.006,{roll},{pitch},{yaw}], a=0.1, v=0.1)\n"
    f"  movel(p[-0.297,-0.546,-0.006,{roll},{pitch},{yaw}], a=3.5, v=3.5, r=0.1)\n"
    f"  movel(p[-0.097,-0.546,-0.006,{roll},{pitch},{yaw}], a=0.5, v=0.5, r=0.5)\n"
    "  movej([-2.095,-2.618,-1.745,1.22,0.5236,-2.618],0.2,0.2)\n"
    "end\n"
)
cmd.sendall(prog.encode("utf-8"))
time.sleep(10)
cmd.close()

# 3) stop & output
ur.stop_logging()
ur.plot("q",   pi_axis=True,  save=True, show=True)
ur.plot("dq",  pi_axis=False, save=True, show=False)
ur.plot("tcp", pi_axis=False, save=True, show=False)
ur.plot("dtcp", pi_axis=False, save=True, show=False)
ur.plot_tcp_xy(save=True, show=True)  # XY path with equal axes (square)
ur.plot_tcp_xyz(save=True, show=True)        # 3D path (equal axis)
print("CSV all:", ur.save_csv(("q","dq","tcp","dtcp","tcp_xy","tcp_xyz")))
ur.close()
