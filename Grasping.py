from pymata_aio.pymata3 import PyMata3
from pymata_aio.constants import Constants

#########################################
# Upload StandardFirmata.ino to Arduino #
#########################################

# Open serial connection
windows_port = 'COM8'
linux_port = '/dev/ttyUSB0'
board = PyMata3(com_port=windows_port)

#print(f"Constants.INPUT: {Constants.INPUT}, Constants.OUTPUT: {Constants.OUTPUT}, Constants.ANALOG: {Constants.ANALOG}, Constants.PWM: {Constants.PWM}")

def check_pins():
    # List of all digital and analog pins for Arduino Due
    digital_pins = list(range(54))  # Digital pins 0-53
    analog_pins = list(range(12))   # Analog pins A0-A11


    # Checking Digital Pins (0-53)
    print("Checking Digital Pins:")
    for pin in digital_pins:
        try:
            board.set_pin_mode(pin, Constants.OUTPUT)  # Set as output mode
            board.digital_write(pin, 1)  # Set HIGH
            print(f"Digital Pin {pin}: OK")
            board.digital_write(pin, 0)  # Set LOW
        except Exception as e:
            print(f"Digital Pin {pin}: Not accessible ({e})")

    # Checking Analog Pins (A0-A11)
    print("\nChecking Analog Pins:")
    for pin in analog_pins:
        try:
            board.set_pin_mode(pin, Constants.ANALOG)  # Set as input mode
            print(f"Analog Pin A{pin}: OK")
        except Exception as e:
            print(f"Analog Pin A{pin}: Not accessible ({e})")

# --- Pin Declaration --- 
# Push Buttons
PushB1_pin = 50
PushB2_pin = 52
PushB3_pin = 48
PushB4_pin = 46
PushB5_pin = 44 

# Potentiometers
POT1_pin = 0
POT2_pin = 1

# Compressor
COMP_pin = 12

# Pressure sensors
PS1_pin = 2
PS2_pin = 3

# Servos
Servo1_pin = 2 # connect to M7 of Exoskeleton (Flexion)
Servo2_pin = 3
Servo3_pin = 4 # connect to M8 of Exoskeleton (Extension)
Servo4_pin = 5
# Define angles for servo motors
Servo1_OUTLET = 110
Servo1_INLET = 20
Servo1_HOLD = 65
Servo2_OUTLET = 130
Servo2_INLET = 40
Servo2_HOLD = 85
Servo3_OUTLET = 130
Servo3_INLET = 30
Servo3_HOLD = 80
Servo4_OUTLET = 130
Servo4_INLET = 40
Servo4_HOLD = 85



def init():
    global PushB1_val_old, PushB2_val_old, PushB3_val_old, PushB4_val_old, PushB5_val_old

    # Potentiometers
    board.set_pin_mode(POT1_pin, Constants.ANALOG)
    board.set_pin_mode(POT2_pin, Constants.ANALOG)

    # Compressor
    board.set_pin_mode(COMP_pin, Constants.PWM)

    # Pressure sensors
    board.set_pin_mode(PS1_pin, Constants.ANALOG)
    board.set_pin_mode(PS2_pin, Constants.ANALOG)

    # Push Buttons
    board.set_pin_mode(PushB1_pin, Constants.INPUT)
    board.set_pin_mode(PushB2_pin, Constants.INPUT)
    board.set_pin_mode(PushB3_pin, Constants.INPUT)
    board.set_pin_mode(PushB4_pin, Constants.INPUT)
    board.set_pin_mode(PushB5_pin, Constants.INPUT)

    PushB1_val_old = board.digital_read(PushB1_pin)
    PushB2_val_old = board.digital_read(PushB2_pin)
    PushB3_val_old = board.digital_read(PushB3_pin)
    PushB4_val_old = board.digital_read(PushB4_pin)
    PushB5_val_old = board.digital_read(PushB5_pin)

    # Put servos in outlet position
    board.servo_config(Servo1_pin)
    board.servo_config(Servo2_pin)
    board.servo_config(Servo3_pin)
    board.servo_config(Servo4_pin)
    board.analog_write(Servo1_pin, Servo1_OUTLET)
    board.analog_write(Servo2_pin, Servo2_OUTLET)
    board.analog_write(Servo3_pin, Servo3_OUTLET)
    board.analog_write(Servo4_pin, Servo4_OUTLET)
    board.sleep(1)
    # Reset servo pins
    board.set_pin_mode(Servo1_pin, Constants.INPUT)
    board.set_pin_mode(Servo2_pin, Constants.INPUT)
    board.set_pin_mode(Servo3_pin, Constants.INPUT)
    board.set_pin_mode(Servo4_pin, Constants.INPUT)
 

def main():
    global PushB1_val_old, PushB2_val_old, PushB3_val_old, PushB4_val_old, PushB5_val_old
    
    while True:
        try:
            # Potentiometers
            POT1_val = board.analog_read(POT1_pin)
            POT2_val = board.analog_read(POT2_pin)
            POT1_voltage = POT1_val * 5 / 1023
            POT2_voltage = POT2_val * 5 / 1023

            # Compressor
            if POT2_voltage >= 5.0:
                board.servo_config(Servo1_pin) # Flexion
                board.servo_config(Servo3_pin) # Extention
                board.analog_write(Servo1_pin, Servo1_INLET)
                board.analog_write(Servo3_pin, Servo3_INLET)
                board.sleep(2)

                # 5 voltage levels related to 5 Purkinje cells
                if POT1_voltage >= 2.5 and POT1_voltage < 3.0:
                    board.analog_write(COMP_pin, int(3.0 * 255 / 5))
                elif POT1_voltage >= 3.0 and POT1_voltage < 3.5:
                    board.analog_write(COMP_pin, int(3.5 * 255 / 5))
                elif POT1_voltage >= 3.5 and POT1_voltage < 4.0:
                    board.analog_write(COMP_pin, int(4.0 * 255 / 5))
                elif POT1_voltage >= 4.0 and POT1_voltage < 4.5:
                    board.analog_write(COMP_pin, int(4.5 * 255 / 5))
                elif POT1_voltage >= 4.5 and POT1_voltage <= 5.0:
                    board.analog_write(COMP_pin, int(5.0 * 255 / 5))
                board.sleep(0.5)
                board.analog_write(Servo1_pin, Servo1_HOLD)
                board.sleep(2.5)
                board.analog_write(Servo3_pin, Servo3_HOLD)
                board.analog_write(COMP_pin, 0)

            else:
                board.analog_write(COMP_pin, 0)
            

            # Pressure sensor
            PS1_val = board.analog_read(PS1_pin)
            PS2_val = board.analog_read(PS2_pin)
            PS1_voltage = PS1_val * 5 / 1023
            PS2_voltage = PS2_val * 5 / 1023

            # Push Buttons
            PushB1_val = board.digital_read(PushB1_pin)
            PushB2_val = board.digital_read(PushB2_pin)
            PushB3_val = board.digital_read(PushB3_pin)
            PushB4_val = board.digital_read(PushB4_pin)
            PushB5_val = board.digital_read(PushB5_pin)

            if PushB1_val != PushB1_val_old:
                board.servo_config(Servo1_pin)
                if PushB1_val == 0:
                    board.analog_write(Servo1_pin, Servo1_INLET)
                    board.sleep(0.5)
                if PushB1_val == 1:
                    board.analog_write(Servo1_pin, Servo1_OUTLET)
                    board.sleep(0.5)
                PushB1_val_old = PushB1_val
            
            if PushB2_val != PushB2_val_old:
                board.servo_config(Servo1_pin)
                if PushB2_val == 1:
                    board.analog_write(Servo1_pin, Servo1_HOLD)
                    board.sleep(0.5)
                PushB2_val_old = PushB2_val
            
            if PushB3_val != PushB3_val_old:
                board.servo_config(Servo3_pin)
                if PushB3_val == 0:
                    board.analog_write(Servo3_pin, Servo3_INLET)
                    board.sleep(0.5)
                if PushB3_val == 1:
                    board.analog_write(Servo3_pin, Servo3_OUTLET)
                    board.sleep(0.5)
                PushB3_val_old = PushB3_val
            
            if PushB4_val != PushB4_val_old:
                board.servo_config(Servo3_pin)
                if PushB4_val == 1:
                    board.analog_write(Servo3_pin, Servo3_HOLD)
                    board.sleep(0.5)
                    board.sleep(0.5)
                PushB4_val_old = PushB4_val

            if PushB5_val != PushB5_val_old:
                if PushB5_val == 0:
                    # Reset servo pins
                    board.analog_write(Servo1_pin, Servo1_OUTLET)
                    board.analog_write(Servo2_pin, Servo2_OUTLET)
                    board.analog_write(Servo3_pin, Servo3_OUTLET)
                    board.analog_write(Servo4_pin, Servo4_OUTLET)
                    board.sleep(1)
                    board.set_pin_mode(Servo1_pin, Constants.INPUT)
                    board.set_pin_mode(Servo2_pin, Constants.INPUT)
                    board.set_pin_mode(Servo3_pin, Constants.INPUT)
                    board.set_pin_mode(Servo4_pin, Constants.INPUT)
                PushB5_val_old = PushB5_val

            print(f"PB1: {PushB1_val} PB2: {PushB2_val} PB3: {PushB3_val} PB4: {PushB4_val} PB5: {PushB5_val} POT1: {POT1_voltage:.2f} POT2: {POT2_voltage:.2f} PS1: {PS1_voltage:.2f}V PS2: {PS2_voltage:.2f}V")

            board.sleep(0.1)  # Small delay to avoid spamming

        except KeyboardInterrupt:
            board.shutdown()
            break

init()
main()





