import serial.tools.list_ports
ports = serial.tools.list_ports.comports()
for port in ports:
    print(f"Device: {port.device}")
    print(f"Name: {port.name}")
    print(f"Description: {port.description}")
    print(f"Hardware ID: {port.hwid}")
    if port.vid and port.pid:
        print(f"  VID:PID: {port.vid:04X}:{port.pid:04X}")
    print("-" * 20)
