import time
from pyfirmata import Arduino, util


# Open serial connection
windows_port = 'COM8'
linux_port = '/dev/ttyUSB0'
board = Arduino(windows_port)

# Start iterator to prevent serial buffer overflow
it = util.Iterator(board)
it.start()
#time.sleep(2) 


DIGITAL = "d"
ANALOG = "a"
INPUT = "i"
OUTPUT = "o"
PWM = "p"

# Push Buttons
PushB_type = DIGITAL
PushB_mode = INPUT
PushB1_pin = 50
PushB2_pin = 52
PushB3_pin = 48
PushB4_pin = 46
PushB5_pin = 44 #8
#PushB5 = board.get_pin(f"{PushB_type}:{PushB5_pin}:{PushB_mode}")
#PushB5.enable_reporting() # Enable reading from the pin


# Potentiometers
POT_type = ANALOG
POT_mode = INPUT
POT1_pin = 0
POT2_pin = 1
POT1 = board.get_pin(f"{POT_type}:{POT1_pin}:{POT_mode}")
POT1.enable_reporting()
POT2 = board.get_pin(f"{POT_type}:{POT2_pin}:{POT_mode}")
POT1.enable_reporting()

# Compressor
COMP_pin = 9 # 12
COMP_type = DIGITAL
COMP_mode = PWM#PWM
COMP = board.get_pin(f"{COMP_type}:{COMP_pin}:{COMP_mode}")




COMP.write(0)


print(f"POT1 {POT1}, POT2 {POT2}")
#print(f"Button5 {PushB5}")
print(f"COMP {COMP}")


while True:
    try:
        #button_state = PushB5.read()
        POT1_val = POT1.read()
        POT2_val = POT2.read()
        if POT1_val is not None:
            COMP.write(POT1_val*255)
 
        if POT1_val is not None and POT2_val is not None:
            print(f"POT1: {POT1_val*255}    POT2: {POT2_val*255}")

        time.sleep(1)  # Small delay to avoid spamming

    except KeyboardInterrupt:
        print("Exiting...")
        board.exit()
        break



