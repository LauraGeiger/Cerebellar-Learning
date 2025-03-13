import serial

# Open the serial connection to ESP32 (adjust COMx port or /dev/ttyUSBx)
ser = serial.Serial('COM11', 115200, timeout=1)  # Adjust port if needed

try:
    while True:
        line = ser.readline().decode().strip()  # Read a line and decode it
        if line:
            sensor_values = list(map(int, line.split(',')))  # Convert CSV to list of integers
            #print("Sensor Values:", sensor_values)
            print(f"R1={sensor_values[0]} L1={sensor_values[1]} R2={sensor_values[2]} L2={sensor_values[3]} R3={sensor_values[4]} L3={sensor_values[5]} R4={sensor_values[6]} L4={sensor_values[7]} R5={sensor_values[8]} L5={sensor_values[9]} T1={sensor_values[10]} T2={sensor_values[11]} ")
except KeyboardInterrupt:
    print("Exiting...")
    ser.close()
