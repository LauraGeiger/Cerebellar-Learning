import serial

# Open the serial connection to ESP32 (adjust COMx port or /dev/ttyUSBx)
ser = serial.Serial('COM11', 115200, timeout=1)  # Adjust port if needed

try:
    while True:
        line = ser.readline().decode().strip()  # Read a line and decode it
        if line:
            sensor_values = list(map(int, line.split(',')))  # Convert CSV to list of integers
            [R1, L1, R2, L2, R3, L3, R4, L4, R5, L5, T1, T2] = sensor_values
            print(f"R1={R1} L1={L1} R2={R2} L2={L2} R3={R3} L3={L3} R4={R4} L4={L4} R5={R5} L5={L5} T1={T1} T2={T2} ")
except KeyboardInterrupt:
    print("Exiting...")
    ser.close()
