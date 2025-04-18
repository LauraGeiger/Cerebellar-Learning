from neuron import h, gui
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.widgets import Button
from matplotlib.widgets import RadioButtons
from matplotlib.gridspec import GridSpec
import time
import serial

from pymata_aio.pymata3 import PyMata3
from pymata_aio.constants import Constants

# using Python 3.10.16

# --- Granule, Purkinje, Inferior Olive, and Basket Cell Classes ---
class GranuleCell:
    def __init__(self, gid):
        self.gid = gid
        self.soma = h.Section(name=f'granule_{gid}')
        self.soma.L = self.soma.diam = 10
        self.soma.insert('hh')

class PurkinjeCell:
    def __init__(self, gid):
        self.gid = gid
        self.soma = h.Section(name=f'purkinje_{gid}')
        self.soma.L = self.soma.diam = 50
        self.soma.insert('hh')

class InferiorOliveCell:
    def __init__(self, gid):
        self.gid = gid
        self.soma = h.Section(name=f'inferior_olive_{gid}')
        self.soma.L = self.soma.diam = 20
        self.soma.insert('hh')

class BasketCell:
    def __init__(self, gid):
        self.gid = gid
        self.soma = h.Section(name=f'basket_{gid}')
        self.soma.L = self.soma.diam = 20
        self.soma.insert('hh')

def init_variables(reset_all=True):
    """Initialize global variables"""
    global fig, gs, ax_network, ax_plots, gs_buttons, animations, purkinje_drawing
    global iter, buttons, state, mode, control_HW, control, mode_dict, hw_dict, control_dict, state_dict, state_grasp_hold_dict, DCN_names
    global colors_purkinje, color_granule, color_inferior_olive, color_basket, color_dcn, color_simple_spike, color_complex_spike, color_error, color_error_hover, color_run, color_run_hover
    global height, width, granule_x, purkinje_x, olive_x, basket_x, dcn_x, granule_y, purkinje_y, olive_y, basket_y, dcn_y
    global t, granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, v_granule, v_purkinje, v_inferiorOlive, v_basket, t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np
    global num_granule, num_purkinje, num_inferior_olive, num_basket, num_dcn, granule_cells, purkinje_cells, inferior_olive_cells, basket_cells
    global pf_syns, pf_ncs, cf_syns, cf_ncs, inh_syns, inh_ncs
    global weights, weights_over_time, pf_initial_weight, cf_initial_weight, basket_initial_weight, stimuli, frequency, processed_GC_spikes, processed_pairs, errors
    global tau_plus, tau_minus, A_plus, A_minus
    global board, pc_air_pressure_mapping, pc_inflation_time_mapping, serial_con, hold_pressure_dict
    global start_time_learning_grasping, start_time_learning_holding

    # --- Plotting ---
    try: # Reset figure
        for widget in buttons.values():
            widget.disconnect_events()  # Disconnect event listeners
            del widget  # Delete the widget instance
        for ax in fig.get_axes():
            ax.remove()  # Remove the axis
    except Exception: # Create figure
        plt.ion()  # Turn on interactive mode
        fig = plt.figure(layout="constrained", figsize=[11,7])

    gs, ax_network, ax_plots, gs_buttons = None, None, None, None
    animations = []
    purkinje_drawing = []

    # --- GUI and Control ---
    iter = 0
    buttons = {}
    if reset_all == True: state = 2                           # 1: light obj (GC0), 2: medium obj (GC1), 3: heavy obj (GC2) 
    if reset_all == True: mode = 0                            # 0: manual, 1: automatic
    if reset_all == True: control_HW = 0                      # 0: simulation, 1: control HW
    if reset_all == True: control = 0                         # 0: control air pressure, 1: control time of thumb & index finger, 2: control air pressure and time of thumb & index finger, 3: grasp & hold
    mode_dict = {0:"Manual Feedback", 1:"Sensor Feedback"}
    hw_dict = {0:"Simulation", 1:"Control HW"}
    control_dict = {0:"Air pressure", 1:"Inflation time", 2:"Pressure & Time", 3:"Grasp & Hold"}
    state_dict = {1: "Light obj.", 2: "Medium obj.", 3: "Heavy obj."}
    state_grasp_hold_dict = {0: "Grasp obj.", 1: "Grasp & hold light obj.", 2: "Grasp & hold medium obj.", 3: "Grasp & hold heavy obj."}
    DCN_names = [
        ["Air Pressure"], 
        ["Timing Thumb Flexion", "Timing Index Finger Flexion"], 
        ["Air Pressure", "Timing Thumb Flexion", "Timing Index Finger Flexion"], 
        ["Timing Thumb\nOpposition & Extension", "Timing Index Finger\nFlexion", "Timing Index Finger\nExtension", "Timing Flexion\nfor Holding"]
    ]
    
    # --- Colors ---
    colors_purkinje = ["steelblue", "darkorange", "mediumseagreen", "crimson", "gold",
    "dodgerblue", "purple", "sienna", "limegreen", "deeppink",
    "teal", "orangered", "indigo", "royalblue", "darkgoldenrod",
    "firebrick", "darkcyan", "tomato", "slateblue", "darkgreen"]
    color_granule = 'darkgoldenrod'
    color_inferior_olive = 'black'
    color_basket = 'darkgray'
    color_dcn = 'darkgray'
    color_simple_spike = 'gold'
    color_complex_spike = 'lightcoral'
    color_error = 'coral'
    color_error_hover = 'lightsalmon'
    color_run = "lightgreen"
    color_run_hover = "palegreen"

    # --- Animation ---
    height, width = None, None
    granule_x, purkinje_x, olive_x, basket_x, dcn_x = None, None, None, None, None
    granule_y, purkinje_y, olive_y, basket_y, dcn_y = None, None, None, None, None

    # --- Spikes and Voltages for Plotting ---
    t = h.Vector()  # First time initialization
    t_np = None
    granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes = None, None, None, None
    v_granule, v_purkinje, v_inferiorOlive, v_basket = None, None, None, None
    v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np = None, None, None, None

    # --- Create Network ---
    if reset_all == True: num_granule = 3
    if reset_all == True: num_purkinje = 5
    if reset_all == True: num_inferior_olive = 1
    if reset_all == True: num_basket = 1
    if reset_all == True: num_dcn = 1
    granule_cells = [GranuleCell(i) for i in range(num_granule)]
    purkinje_cells = [PurkinjeCell(i) for i in range(num_purkinje)]
    inferior_olive_cells = [InferiorOliveCell(i) for i in range(num_inferior_olive)]
    basket_cells = [BasketCell(i) for i in range(num_basket)]

    # --- Create Synapses and Connections ---
    pf_syns = [[None for _ in range(num_purkinje)] for _ in range(num_granule)] # parallel fiber synapses
    pf_ncs = [[None for _ in range(num_purkinje)] for _ in range(num_granule)] # parallel fiber netcons
    cf_syns = [[None for _ in range(num_purkinje)] for _ in range(num_inferior_olive)] # climbing fiber synapses
    cf_ncs = [[None for _ in range(num_purkinje)] for _ in range(num_inferior_olive)] # climbing fiber netcons
    inh_syns = [[None for _ in range(num_purkinje)] for _ in range(num_basket)] # inhibitory synapses
    inh_ncs = [[None for _ in range(num_purkinje)] for _ in range(num_basket)] # inhibitory netcons

    # --- Spikes and Weights ---
    weights = {}
    weights_over_time = { (pre_gid, post_gid): [] for pre_gid in range(num_granule) for post_gid in range(num_purkinje) } # track weights over time
    pf_initial_weight = 0.02 # Parallel fiber initial weight
    cf_initial_weight = 0.3 # Climbing fiber initial weight
    basket_initial_weight = 0.5 # Basket to Purkinje weight
    stimuli = []
    frequency = 50 # Hz
    processed_GC_spikes = { (g_gid): set() for g_gid in range(num_granule)} # store the processed granule cell spikes
    processed_pairs = { (pre_id, post_id): set() for pre_id in range(num_granule) for post_id in range(num_purkinje) } # store the processed spike pairs for each (pre_id, post_id)
    errors = [False]*num_dcn

    # --- Learning Parameters ---
    tau_plus = 1 
    tau_minus = 1.5
    A_plus = 0.05  
    A_minus = 0.06

    # --- HW Paramaters ---
    # --- Actuator board ---
    if 'board' not in globals():
        board = None # Init board only once
    min_air_pressure = 150 # Minimum PWM value to control air compressor
    max_air_pressure = 255 # Minimum PWM value to control air compressor
    air_pressures = np.linspace(min_air_pressure, max_air_pressure, num_purkinje)
    pc_air_pressure_mapping = {i: air_pressures[i] for i in range(num_purkinje)} # Maps each Purkinje Cell to a specific PWM value to control the air compressor
    min_inflation_time = 1 # Minimum inflation time in seconds
    max_inflation_time = 3 # Maximum inflation time in seconds
    group_size = (num_purkinje + 1) // num_dcn
    inflation_times = np.linspace(min_inflation_time, max_inflation_time, group_size)
    pc_inflation_time_mapping = {i: inflation_times[i % group_size] for i in range(num_purkinje)} # Maps each Purkinje Cell to a specific inflation time to control the valves
    # --- Sensor board ---
    if 'serial_con' not in globals():
        serial_con = None # Init serial connection only once
    hold_pressure_dict = {0: 35, 1: 30, 2: 45, 3: 60} # maximum allowed pressure for grasping (0), holding light object (1), holding medium object (2), holding heavy object (3)  
    start_time_learning_grasping = None
    start_time_learning_holding = None

def init_HW():
    """Initializes Arduino board"""
    global board, PushB1_pin, PushB2_pin, PushB3_pin, PushB4_pin, PushB5_pin, POT1_pin, POT2_pin, COMP_pin, PS1_pin, PS2_pin, PS1_voltage, PS2_voltage, Servo1_pin, Servo2_pin, Servo3_pin, Servo4_pin
    global Servo1_OUTLET, Servo1_INLET, Servo1_HOLD, Servo2_OUTLET, Servo2_INLET, Servo2_HOLD, Servo3_OUTLET, Servo3_INLET, Servo3_HOLD, Servo4_OUTLET, Servo4_INLET, Servo4_HOLD
    global PushB1_val_old, PushB2_val_old, PushB3_val_old, PushB4_val_old, PushB5_val_old 
    #########################################
    # Upload StandardFirmata.ino to Arduino #
    #########################################
    port = "COM8" # change if needed
    # Open serial connection to Arduino
    if board is None:
        board = PyMata3(com_port=port)
    
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
    PS1_voltage = None
    PS2_voltage = None

    # Servos
    Servo1_pin = 2 # connect to Flexion of Thumb
    Servo2_pin = 3 # connect to Flexion of Index Finger 
    Servo3_pin = 4 # connect to Opposition & Extension of Thumb
    Servo4_pin = 5 # connect to Extension of Index Finger
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

    # --- Pin Allocation ---
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
    board.sleep(2)
    # Reset servo pins
    board.set_pin_mode(Servo1_pin, Constants.INPUT)
    board.set_pin_mode(Servo2_pin, Constants.INPUT)
    board.set_pin_mode(Servo3_pin, Constants.INPUT)
    board.set_pin_mode(Servo4_pin, Constants.INPUT)

def create_connections():
    """Create synapses between the cells and initialize weights"""
    global weights

    # Granule → Purkinje Connections (excitatory)
    for purkinje in purkinje_cells:
        random_weight = np.random.uniform(0,0.001)
        for granule in granule_cells:
            syn = h.Exp2Syn(purkinje.soma(0.5))
            syn.e = 0 # Excitatory
            syn.tau1 = 1 # Synaptic rise time
            syn.tau2 = 5 # Synaptic decay time
            pf_syns[granule.gid][purkinje.gid] = syn
            nc = h.NetCon(granule.soma(0.5)._ref_v, syn, sec=granule.soma)
            nc.weight[0] = pf_initial_weight + random_weight
            nc.delay = 0
            pf_ncs[granule.gid][purkinje.gid] = nc
            weights[(granule.gid, purkinje.gid)] = nc.weight[0]
    
    # Inferior Olive → Purkinje Connections (excitatory)
    group_size = num_purkinje // num_inferior_olive  # Size of each IO’s Purkinje group
    remainder = num_purkinje % num_inferior_olive  # Handle remainder case

    for inferior_olive in inferior_olive_cells:
        start_idx = inferior_olive.gid * group_size + min(inferior_olive.gid, remainder)  
        end_idx = start_idx + group_size + (1 if inferior_olive.gid < remainder else 0)  

        for purkinje_gid in range(start_idx, end_idx):
            purkinje = purkinje_cells[purkinje_gid]
            syn = h.Exp2Syn(purkinje.soma(0.5))
            syn.e = 0  # Excitatory
            syn.tau1 = 5 # Synaptic rise time
            syn.tau2 = 25 # Synaptic decay time
            cf_syns[inferior_olive.gid][purkinje.gid] = syn
            nc = h.NetCon(inferior_olive.soma(0.5)._ref_v, syn, sec=inferior_olive.soma)
            nc.weight[0] = 0
            nc.delay = 0
            cf_ncs[inferior_olive.gid][purkinje.gid] = nc

    # Basket → Purkinje Connections (inhibitory)
    for basket in basket_cells:
        for purkinje in purkinje_cells:
            syn = h.Exp2Syn(purkinje.soma(0.5))
            syn.e = -70  # Inhibitory
            syn.tau1 = 1 # Synaptic rise time
            syn.tau2 = 5 # Synaptic decay time
            inh_syns[basket.gid][purkinje.gid] = syn
            nc = h.NetCon(basket.soma(0.5)._ref_v, syn, sec=basket.soma)
            nc.weight[0] = basket_initial_weight
            nc.delay = 0
            inh_ncs[basket.gid][purkinje.gid] = nc

def activate_highest_weight_PC(granule_gid):
    """Activate the PC with the highest weight (that is not blocked)"""
    global inh_syns, inh_ncs, stimuli

    # Initialize dictionaries for max_weights and active Purkinje cells
    max_weights = [-np.inf] * num_dcn
    active_purkinje = [None] * num_dcn

    def get_active_purkinje(purkinje, weight, weight_key):
        if weight > max_weights[weight_key]:
                max_weights[weight_key] = weight
                active_purkinje[weight_key] = purkinje

    # Find the Purkinje cell with the highest weight
    for purkinje in purkinje_cells:
        try:
            if v_purkinje_np[purkinje.gid][-1] > -56: # if membrane voltage is above 55 mV
                continue # Skip the blocked Purkinje cell
        except (NameError, IndexError):
            continue

        weight = pf_ncs[granule_gid][purkinje.gid].weight[0]

        for i in range(num_dcn if state > 0 else num_dcn - 1):
            if purkinje.gid >= i * num_purkinje//num_dcn and purkinje.gid < (i+1) * num_purkinje//num_dcn:
                get_active_purkinje(purkinje, weight, i)

    # Set inhibition and climbing fiber weights for all Purkinje cells
    for purkinje in purkinje_cells:
        is_active = purkinje in active_purkinje # Check if the Purkinje cell is active

        # Set inhibition and climbing fiber weights
        new_inh_weight = 0 if is_active else basket_initial_weight
        new_cf_weight = cf_initial_weight if is_active else 0

        for basket in basket_cells:
            if inh_ncs[basket.gid][purkinje.gid] is not None:
                inh_ncs[basket.gid][purkinje.gid].weight[0] = new_inh_weight
        
        for inferior_olive in inferior_olive_cells:
            if cf_ncs[inferior_olive.gid][purkinje.gid] is not None:
                cf_ncs[inferior_olive.gid][purkinje.gid].weight[0] = new_cf_weight
    
def release_actuator():
    """Releases air from tubes"""
    # Config servo pins
    time.sleep(0.1)
    board.servo_config(Servo1_pin)
    board.servo_config(Servo2_pin) 
    board.servo_config(Servo3_pin)
    board.servo_config(Servo4_pin) 
    time.sleep(0.1)
    # Set servos to outlet position to let air out
    board.analog_write(Servo1_pin, Servo1_OUTLET) # Release flexion thumb
    board.analog_write(Servo2_pin, Servo2_OUTLET) # Release flexion index finger
    board.sleep(1)
    board.analog_write(Servo3_pin, Servo3_OUTLET) # Release opposition & extension thumb 
    board.analog_write(Servo4_pin, Servo4_OUTLET) # Release extension index finger
    board.sleep(1)
    # Reset servo pins
    board.set_pin_mode(Servo1_pin, Constants.INPUT)
    board.set_pin_mode(Servo2_pin, Constants.INPUT)
    board.set_pin_mode(Servo3_pin, Constants.INPUT)
    board.set_pin_mode(Servo4_pin, Constants.INPUT)

def grasp(time_thumb_flexion=1, time_index_flexion=5, time_index_extension = 0.5, time_thumb_opposition=3, pressure=255):
    """Grasp object with variable timing for thumb and index finger flexion and thumb opposition"""

    time_thumb_extension = time_thumb_opposition # Thumb extension equivalent to thumb opposition
    
    print(f"GRASPING: Air pressure: {pressure:.0f} ({int(pressure/255.0*100)}%) Inflation times: Thumb Flexion {time_thumb_flexion:.1f}s Thumb Extension {time_thumb_extension:.1f}s Thumb Opposition {time_thumb_opposition:.1f}s Index Finger Flexion {time_index_flexion:.1f}s Index Finger Extension {time_index_extension:.1f}s")
    control_actuator(pressure=pressure, time_thumb_flexion=time_thumb_flexion, time_index_flexion=time_index_flexion, time_thumb_opposition=time_thumb_opposition, time_thumb_extension=time_thumb_extension, time_index_extension=time_index_extension)
    

def hold(inflation_time=1, pressure=255):
    """Hold object with variable inflation_time, higher inflation_time correlates to higher pressure during holding"""
    
    time_index_flexion = inflation_time
    time_thumb_flexion = inflation_time / 10.0

    print(f"HOLDING: Air pressure: {pressure:.0f} ({int(pressure/255.0*100)}%) Inflation times: Thumb Flexion {time_thumb_flexion:.2f}s Index Finger Flexion {time_index_flexion:.2f}s")
    control_actuator(pressure=pressure, time_index_flexion=time_index_flexion, time_thumb_flexion=time_thumb_flexion)
    

def control_actuator(pressure=None, time_thumb_flexion=None, time_index_flexion=None, time_thumb_opposition=None, time_thumb_extension=None, time_index_extension=None):
    """ Control air compressor and valves via Arduino board"""
    
    time.sleep(0.1) 
    if time_thumb_flexion is not None:
        board.servo_config(Servo1_pin) # Flexion - Thumb
        time.sleep(0.1) 
        board.analog_write(Servo1_pin, Servo1_INLET)
    if time_index_flexion is not None:
        board.servo_config(Servo2_pin) # Flexion - Index Finger
        time.sleep(0.1) 
        board.analog_write(Servo2_pin, Servo2_INLET)
    if time_thumb_opposition is not None or time_thumb_extension is not None:
        board.servo_config(Servo3_pin) # Opposition % Extension - Thumb
        time.sleep(0.1) 
        board.analog_write(Servo3_pin, Servo3_INLET)
    if time_index_extension is not None:
        board.servo_config(Servo4_pin) # Extension - Index finger
        time.sleep(0.1) 
        board.analog_write(Servo4_pin, Servo4_INLET)    
    
    board.sleep(2)

    if pressure is not None:
        board.analog_write(COMP_pin, int(pressure)) # Start inflation

    # Determine the total inflation time
    max(x for x in (1, 2, 3, None) if x is not None)  # Returns 3

    total_inflation_time = max(time for time in(time_thumb_flexion, time_index_flexion, time_thumb_opposition, time_thumb_extension, time_index_extension) if time is not None)

    # Get start time
    start_time = time.time()

    # Flags to track which actions have been executed
    executed = {
        "stop_thumb_extension": False,
        "stop_index_extension": False,
        "stop_thumb_flexion": False,
        "stop_index_flexion": False,
        "stop_thumb_opposition": False
    }

    while True:
        elapsed_time = time.time() - start_time  # Calculate elapsed time

        if time_index_extension is not None and elapsed_time >= time_index_extension and not executed["stop_index_extension"]:
            board.analog_write(Servo4_pin, Servo4_HOLD)  # Stop extension - Index
            executed["stop_index_extension"] = True

        if time_thumb_flexion is not None and elapsed_time >= time_thumb_flexion and not executed["stop_thumb_flexion"]:
            board.analog_write(Servo1_pin, Servo1_HOLD)  # Stop flexion - Thumb
            executed["stop_thumb_flexion"] = True

        if time_index_flexion is not None and elapsed_time >= time_index_flexion and not executed["stop_index_flexion"]:
            board.analog_write(Servo2_pin, Servo2_HOLD)  # Stop flexion - Index
            executed["stop_index_flexion"] = True
        
        if time_thumb_opposition is not None and elapsed_time >= time_thumb_opposition and not executed["stop_thumb_opposition"]:
            board.analog_write(Servo3_pin, Servo3_HOLD)  # Stop opposition & extension - Thumb
            executed["stop_thumb_opposition"] = True
            executed["stop_thumb_extension"] = True

        # Exit loop once all actions are completed
        if elapsed_time >= total_inflation_time:
            break

        time.sleep(0.01)  # Small delay to prevent CPU overload

    board.analog_write(COMP_pin, 0) # Stop inflation

    # Reset servo pins
    if time_thumb_flexion is not None:
        board.set_pin_mode(Servo1_pin, Constants.INPUT)
    if time_index_flexion is not None:
        board.set_pin_mode(Servo2_pin, Constants.INPUT)
    if time_thumb_opposition is not None or time_thumb_extension is not None:
        board.set_pin_mode(Servo3_pin, Constants.INPUT)
    if time_index_extension is not None:
        board.set_pin_mode(Servo4_pin, Constants.INPUT)
   
def stimulate_granule_cell():
    """Stimulate Granule Cells Based on State"""
    
    if state > 0:
        g_ids = [state-1] 
    else:
        g_ids = [g_id for g_id in range(num_granule)]
    
    for g_id in g_ids:
        stim = h.IClamp(granule_cells[g_id].soma(0.5))
        stim.delay = 1/frequency*1000 * (iter + 1/3)
        stim.dur = 1
        stim.amp = 0.5
        stimuli.append(stim)

    # Send inhibitory signals to all purkinje cells expect active_purkinje
    for basket in basket_cells:
        basket_stim = h.IClamp(basket.soma(0.5))
        basket_stim.delay = stim.delay # same as granule 
        basket_stim.dur = stim.dur # same as granule 
        basket_stim.amp = stim.amp  # same as granule 
        stimuli.append(basket_stim)

def change_back_error_button_colors():
    """Changes the color of all error buttons back to default color"""
    global buttons
    # Change color of error buttons to default color
    try:
        buttons["error_button"].color = "0.85"
        buttons["error_button"].hovercolor = "0.975"
        buttons["error_button"].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    try:
        buttons["error_pressure"].color = "0.85"
        buttons["error_pressure"].hovercolor = "0.975"
        buttons["error_pressure"].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    try:
        buttons["error_thumb"].color = "0.85"
        buttons["error_thumb"].hovercolor = "0.975"
        buttons["error_thumb"].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    try:
        buttons["error_index"].color = "0.85"
        buttons["error_index"].hovercolor = "0.975"
        buttons["error_index"].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    try:
        buttons["error_index_flexion"].color = "0.85"
        buttons["error_index_flexion"].hovercolor = "0.975"
        buttons["error_index_flexion"].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    try:
        buttons["error_index_extension"].color = "0.85"
        buttons["error_index_extension"].hovercolor = "0.975"
        buttons["error_index_extension"].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    try:
        buttons["error_opposition"].color = "0.85"
        buttons["error_opposition"].hovercolor = "0.975"
        buttons["error_opposition"].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    try:
        buttons["error_holding"].color = "0.85"
        buttons["error_holding"].hovercolor = "0.975"
        buttons["error_holding"].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(1)

def update_granule_stimulation_and_plots(event=None):
    """Stimulates one granule cell and updates the plots and controls the HW (if enabled)"""
    global granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, buttons, iter, ax_network, animations
    global errors, start_time_learning_grasping, start_time_learning_holding

    if start_time_learning_grasping == None:
        start_time_learning_grasping = time.time()
    if state != 0 and start_time_learning_holding == None:
        start_time_learning_holding = time.time()

    # Apply errors
    for i, error in enumerate(errors):
        if error:
            update_inferior_olive_stimulation_and_plots(cell_nr=i)

    # Reset errors
    errors = [False]*num_dcn

    try: # change back color of "grasp successul" button
        buttons["success_grasp"].color = "0.85"
        buttons["success_grasp"].hovercolor = "0.975"
        buttons["success_grasp"].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    change_back_error_button_colors()
    time.sleep(0.1)
    
    run_simulation(error=True)

    # Activate PC with highest weight
    g_id = (state - 1) % num_granule # choose one GC for calculation and plotting
    activate_highest_weight_PC(g_id)

    # Identify active purkinje cells
    b_id = 0
    p_ids = []
    if control == 0: # Control air pressure
        p_ids.append(next((purkinje.gid for purkinje in purkinje_cells if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None))
    elif control == 1: # Control inflation time
        p_ids.append(next((purkinje.gid for purkinje in purkinje_cells[:num_purkinje//2] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None))
        p_ids.append(next((purkinje.gid for purkinje in purkinje_cells[num_purkinje//2:] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None))
    elif control == 2: # Control air pressure & inflation time
        p_ids.append(next((purkinje.gid for purkinje in purkinje_cells[:num_purkinje//3] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None))
        p_ids.append(next((purkinje.gid for purkinje in purkinje_cells[num_purkinje//3:2*num_purkinje//3] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None))
        p_ids.append(next((purkinje.gid for purkinje in purkinje_cells[2*num_purkinje//3:] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None))
    elif control == 3: # Control grasp & hold
        p_ids.append(next((purkinje.gid for purkinje in purkinje_cells[:num_purkinje//4] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None))
        p_ids.append(next((purkinje.gid for purkinje in purkinje_cells[num_purkinje//4:2*num_purkinje//4] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None))
        p_ids.append(next((purkinje.gid for purkinje in purkinje_cells[2*num_purkinje//4:3*num_purkinje//4] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None))
        if state > 0:
            p_ids.append(next((purkinje.gid for purkinje in purkinje_cells[3*num_purkinje//4:] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None))

    stimulate_granule_cell()
    run_simulation()
    iter += 1
    buttons["run_button"].label.set_text(f"Run iteration {iter}")
    update_spike_and_weight_plot()

    # --- Control of actuator ---
    if control_HW:
        release_actuator()

        board.sleep(1)

        if control == 0: # Control air pressure
            air_pressure = pc_air_pressure_mapping[p_ids[0]] if p_ids[0] is not None else 0 # look up table for purkinje cell to voltage mapping
            grasp(pressure=air_pressure)
        elif control == 1: # Control inflation time
            inflation_time_thumb = pc_inflation_time_mapping[p_ids[0]] if p_ids[0] is not None else 0 # look up table for purkinje cell to time mapping
            inflation_time_index = pc_inflation_time_mapping[p_ids[1]] if p_ids[1] is not None else 0 # look up table for purkinje cell to time mapping
            grasp(time_thumb_flexion=inflation_time_thumb, time_index_flexion=inflation_time_index)
        elif control == 2: # Control air pressure & inflation time
            air_pressure = pc_air_pressure_mapping[p_ids[0]] if p_ids[0] is not None else 0 # look up table for purkinje cell to voltage mapping
            inflation_time_thumb = pc_inflation_time_mapping[p_ids[1]] if p_ids[1] is not None else 0 # look up table for purkinje cell to time mapping
            inflation_time_index = pc_inflation_time_mapping[p_ids[2]] if p_ids[2] is not None else 0 # look up table for purkinje cell to time mapping
            grasp(pressure=air_pressure, time_thumb_flexion=inflation_time_thumb, time_index_flexion=inflation_time_index)
        elif control == 3: # Control time for flexion and opposition
            time_thumb_opposition = pc_inflation_time_mapping[p_ids[0]] if p_ids[0] is not None else 0 # look up table for purkinje cell to time mapping
            time_index_flexion = pc_inflation_time_mapping[p_ids[1]] if p_ids[1] is not None else 0 # look up table for purkinje cell to time mapping
            time_index_extension = pc_inflation_time_mapping[p_ids[2]] if p_ids[2] is not None else 0 # look up table for purkinje cell to time mapping
            grasp(time_thumb_opposition=time_thumb_opposition, time_index_flexion=time_index_flexion, time_index_extension=time_index_extension)
        
        board.sleep(1)

    if buttons["network_button"].label.get_text() == "Hide network":
        # Run simple spike animation
        spikes = []
        if state > 0:
            g_ids = [state-1] 
        else:
            g_ids = [g_id for g_id in range(num_granule)]
        

        for g in g_ids:
            for p in p_ids:
                spike, = ax_network.plot([], [], marker='o', color=color_simple_spike, markersize=10)
                spikes.append(spike)
        ani = animation.FuncAnimation(ax_network.figure, update_animation, frames=60, interval = 20, blit=True, repeat=False, fargs=(spikes, 0, p_ids*len(g_ids), g_ids*len(p_ids)))
        animations.append(ani)
        plt.pause(5)
        #time.sleep(2)
        update_weights_in_network()        

    # --- Trigger error based on sensor feedback ---
    if mode == 1:
        serial_con.flushInput() # delete values in serial input buffer
        #print("Waiting for sensor feedback ...")

        timeout = 5 # timeout after 5 seconds
        grasping = False
        holding_time = 5 # grasping successfull if object was holded for 5 seconds

        start_time = time.time()

        while True:
            remaining_time = timeout - (time.time() - start_time)

            if remaining_time < 0:
                if not any(errors): # If no error feedback is received before timeout, trigger randomly any error
                    if control == 0:
                        random_cell_nr = np.random.randint(control + 1)
                        error_detected(btn_name="error_button", cell_nr=random_cell_nr)
                        print(f"\nTimeout: triggering error for {DCN_names[control][random_cell_nr]}")
                    elif control == 1:
                        random_cell_nr = np.random.randint(control + 1)
                        btn_name = "error_thumb" if random_cell_nr == 0 else "error_index"
                        error_detected(btn_name=btn_name, cell_nr=random_cell_nr)
                        print(f"\nTimeout: triggering random error for {DCN_names[control][random_cell_nr]}")
                    elif control == 2:
                        random_cell_nr = np.random.randint(control + 1)
                        if random_cell_nr == 0:
                            btn_name = "error_pressure"
                        elif random_cell_nr == 1:
                            btn_name = "error_thumb"
                        else:
                            btn_name = "error_index"
                        error_detected(btn_name=btn_name, cell_nr=random_cell_nr)
                        print(f"\nTimeout: triggering radnom error for {DCN_names[control][random_cell_nr]}")
                    elif control == 3:
                        random_cell_nr = np.random.randint(control)
                        if random_cell_nr == 0:
                            btn_name = "error_opposition"
                        elif random_cell_nr == 1:
                            btn_name = "error_index_flexion"
                        else:
                            btn_name = "error_index_extension"
                        error_detected(btn_name=btn_name, cell_nr=random_cell_nr)
                        print(f"\nTimeout: triggering random error for {DCN_names[control][random_cell_nr]}")
                break
            else:
                try:
                    line = serial_con.readline().decode(errors='ignore').strip()  # Read a line and decode it
                    while line and line.count(',') < 15:
                        line = serial_con.readline().decode(errors='ignore').strip()  # Read a line and decode it
                        time.sleep(0.1) # small delay to wait for new sensor values
                    if line:
                        sensor_values = list(map(int, line.split(',')))  # Convert CSV to list of integers
                        [FS_I1, FS_T1, FS_I2, FS_T2, FS_I3, FS_T3, FS_I4, FS_T4, FS_I5, FS_T5, TS_T, TS_I, STS_T, STS_P, STS_M, STS_I] = sensor_values # Flexsensors: FS_x, Touchsensors: TS_x, SoftTouchsensors: STS_x , T (Thumb), P (Palm), M (Middle Finger), I (Index Finger)
                        print(f"\rRemaining time: {remaining_time:.0f}s | FS_I1={FS_I1} FS_T1={FS_T1} FS_I2={FS_I2} FS_T2={FS_T2} FS_I3={FS_I3} FS_T3={FS_T3} FS_I4={FS_I4} FS_T4={FS_T4} FS_I5={FS_I5} FS_T5={FS_T5} TS_T={TS_T} TS_I={TS_I} STS_T={STS_T} STS_I={STS_I}", end="", flush=True)
                        applied_pressure = (TS_T + TS_I) / 2.0
                        if (TS_T > 0 and TS_I > 0) and applied_pressure <= hold_pressure_dict[0]: # Object successful grasped
                            time_learning_grasping = time.time() - start_time_learning_grasping
                            start_time_learning_grasping = None
                            print(f"\nGrasping successfull, applied pressure={applied_pressure:.1f}%, required time={time_learning_grasping:.0f}s")
                            grasping = True
                            grasp_successfull()
                            break
                        elif TS_T > 0 and TS_I > 0: # Object grasped but too much pressure applied
                            print(f"\nObject grasped but too much pressure applied ({applied_pressure:.1f}%>{hold_pressure_dict[0]}%).")
                            if control == 3:
                                random_cell_nr = np.random.randint(1, control)
                                print(f"\nTriggering error for index finger, randomly selected {DCN_names[control][random_cell_nr]} (PC{p_ids[random_cell_nr]+1})")
                                btn_name = "error_index_flexion" if random_cell_nr == 1 else "error_index_extension"
                                error_detected(btn_name=btn_name, cell_nr=random_cell_nr)
                            break
                        else: # Grasping not successfull
                            if control == 0:
                                if errors[0] == False and any (value > 0 for value in [FS_T1, FS_I1, FS_T2, FS_I2, FS_T3, FS_I3, FS_T4, FS_I4, FS_T5, FS_I5]): # Grasping not successfull
                                    print(f"\nDetected error for {DCN_names[control][0]} (PC{p_ids[0]+1})")
                                    error_detected(btn_name="error_button", cell_nr=0)

                            elif control == 1:
                                if errors[0] == False and any (value > 0 for value in [FS_I1, FS_I2, FS_I3, FS_I4, FS_I5]): # Error in thumb movement
                                    print(f"\nDetected error for {DCN_names[control][0]} (PC{p_ids[0]+1})")
                                    error_detected(btn_name="error_thumb", cell_nr=0)
                                if errors[1] == False and any (value > 0 for value in [FS_T1, FS_T2, FS_T3, FS_T4, FS_T5]): # Error in index finger movement
                                    print(f"\nDetected error for {DCN_names[control][1]} (PC{p_ids[1]+1})")
                                    error_detected(btn_name="error_index", cell_nr=1)
                            
                            elif control == 2:
                                if errors[1] == False and any (value > 0 for value in [FS_I1, FS_I2, FS_I3, FS_I4, FS_I5]): # Error in thumb movement
                                    print(f"\nDetected error for {DCN_names[control][1]} (PC{p_ids[1]+1})")
                                    error_detected(btn_name="error_thumb", cell_nr=1)
                                if errors[2] == False and any (value > 0 for value in [FS_T1, FS_T2, FS_T3, FS_T4, FS_T5]): # Error in index finger movement
                                    print(f"\nDetected error for {DCN_names[control][2]} (PC{p_ids[2]+1})")
                                    error_detected(btn_name="error_index", cell_nr=2)

                            elif control == 3:
                                if state == 0:
                                    if errors[0] == False and (TS_I > 0 or any (valueR > 0 for valueR in [FS_I1, FS_I2, FS_I3, FS_I4, FS_I5])): # Index finger correct, Error in thumb opposition
                                        print(f"\nDetected error for {DCN_names[control][0]} (PC{p_ids[0]+1})")
                                        error_detected(btn_name="error_opposition", cell_nr=0)
                                    if errors[1] == False and errors[2] == False and (TS_T > 0 or any (value > 0 for value in [FS_T1, FS_T2, FS_T3, FS_T4, FS_T5])): # Thumb correct, Error in index finger 
                                        random_cell_nr = np.random.randint(1, control)
                                        print(f"\nDetected error for index finger, randomly selected {DCN_names[control][random_cell_nr]} (PC{p_ids[random_cell_nr]+1})")
                                        btn_name = "error_index_flexion" if random_cell_nr == 1 else "error_index_extension"
                                        error_detected(btn_name=btn_name, cell_nr=random_cell_nr)
                            
                except Exception: None

            serial_con.flushInput() # delete values in serial input buffer
            time.sleep(0.1) # small delay to prevent CPU overload

        if grasping:
            if state != 0: # 0: grasp only, 1,2,3: grasp & hold
                if control == 3:
                    if control_HW == 1:
                        time_flexion_holding = pc_inflation_time_mapping[p_ids[-1]] if p_ids[-1] is not None else 0 # look up table for purkinje cell to voltage mapping
                        time_flexion_holding -= pc_inflation_time_mapping[0]
                        hold(inflation_time=time_flexion_holding)
                try:
                    line = serial_con.readline().decode(errors='ignore').strip()  # Read a line and decode it
                    while line and line.count(',') < 15:
                        line = serial_con.readline().decode(errors='ignore').strip()  # Read a line and decode it
                        time.sleep(0.1) # small delay to wait for new sensor values
                    if line:
                        sensor_values = list(map(int, line.split(',')))  # Convert CSV to list of integers
                        [FS_I1, FS_T1, FS_I2, FS_T2, FS_I3, FS_T3, FS_I4, FS_T4, FS_I5, FS_T5, TS_T, TS_I, STS_T, STS_P, STS_M, STS_I] = sensor_values # Flexsensors: FS_x, Touchsensors: TS_x, SoftTouchsensors: STS_x , T (Thumb), P (Palm), M (Middle Finger), I (Index Finger)
                        applied_pressure = 0
                        if TS_T > 0 or TS_I > 0: # Object detected with touch sensors
                            applied_pressure = (TS_T + TS_I) / 2.0
                        print(f"Ready for holding, applied pressure={applied_pressure:.1f}%")
                except Exception: None
                
                start_time_holding = time.time()
                print("\n")
                while True:
                    remaining_time = holding_time - (time.time() - start_time_holding)

                    if remaining_time < 0:
                        applied_pressure = (TS_T + TS_I) / 2.0
                        if (TS_T > 0 or TS_I > 0) and applied_pressure <= hold_pressure_dict[state]: # Object successful holded
                            time_learning_holding = time.time() - start_time_learning_holding
                            start_time_learning_holding = None
                            print(f"\nHolding successful, applied pressure={applied_pressure:.1f}%, required time={time_learning_holding:.0f}s")
                            print(f"FS_I1={FS_I1} FS_T1={FS_T1} FS_I2={FS_I2} FS_T2={FS_T2} FS_I3={FS_I3} FS_T3={FS_T3} FS_I4={FS_I4} FS_T4={FS_T4} FS_I5={FS_I5} FS_T5={FS_T5} TS_T={TS_T} TS_I={TS_I} STS_T={STS_T} STS_I={STS_I}")
                            grasping = True
                            errors = [False for _ in errors] # reset all errors
                        elif TS_T > 0 or TS_I > 0: # Object holded but too much pressure applied
                            print(f"\nObject holded but too much pressure applied ({applied_pressure:.1f}%>{hold_pressure_dict[state]}%).")
                            if control == 3:
                                print(f"Triggering error for {DCN_names[control][3]} (PC{p_ids[3]+1})")
                                error_detected(btn_name="error_holding", cell_nr=3)
                        else: # Object not holded
                            print(f"\nObject grasped but not holded for {holding_time}s.")
                            if control == 0: # if air pressure is controlled
                                print(f"Triggering error for {DCN_names[control][0]} (PC{p_ids[0]+1})")
                                error_detected(btn_name="error_button", cell_nr=0)
                            elif control == 1: # if air pressure is constant
                                print(f"Triggering error for {DCN_names[control][0]} (PC{p_ids[0]+1}) and {DCN_names[control][1]} (PC{p_ids[1]+1})")
                                error_detected(btn_name="error_thumb", cell_nr=0)
                                error_detected(btn_name="error_index", cell_nr=1)
                            elif control == 2: # if air pressure is controlled
                                print(f"Triggering error for {DCN_names[control][0]} (PC{p_ids[0]+1})")
                                error_detected(btn_name="error_pressure", cell_nr=0)
                            elif control == 3: # holding pressure controlled via timing
                                print(f"Triggering error for {DCN_names[control][3]} (PC{p_ids[3]+1})")
                                error_detected(btn_name="error_holding", cell_nr=3)
                        break
                    else:
                        try:
                            line = serial_con.readline().decode(errors='ignore').strip()  # Read a line and decode it
                            while line and line.count(',') < 15:
                                line = serial_con.readline().decode(errors='ignore').strip()  # Read a line and decode it
                                time.sleep(0.1) # small delay to wait for new sensor values
                            if line:
                                sensor_values = list(map(int, line.split(',')))  # Convert CSV to list of integers
                                [FS_I1, FS_T1, FS_I2, FS_T2, FS_I3, FS_T3, FS_I4, FS_T4, FS_I5, FS_T5, TS_T, TS_I, STS_T, STS_P, STS_M, STS_I] = sensor_values # Flexsensors: FS_x, Touchsensors: TS_x, SoftTouchsensors: STS_x , T (Thumb), P (Palm), M (Middle Finger), I (Index Finger)
                                print(f"\rWaiting for: {remaining_time:.0f}s | FS_I1={FS_I1} FS_T1={FS_T1} FS_I2={FS_I2} FS_T2={FS_T2} FS_I3={FS_I3} FS_T3={FS_T3} FS_I4={FS_I4} FS_T4={FS_T4} FS_I5={FS_I5} FS_T5={FS_T5} TS_T={TS_T} TS_I={TS_I} STS_T={STS_T} STS_I={STS_I}", end="", flush=True)
                        except Exception: None

                    serial_con.flushInput() # delete values in serial input buffer
                    time.sleep(0.1) # small delay to prevent CPU overload
        else:
            if state != 0:
                print(f"Grasping not successful, no error will be applied, try again")
                errors = [False for _ in errors] # reset all errors
            time.sleep(0.1)
            update_granule_stimulation_and_plots() # automatically run next iteration until grasping was performed successfully

    buttons["run_button"].color = color_run
    buttons["run_button"].hovercolor = color_run_hover
    buttons["run_button"].ax.figure.canvas.draw_idle()  # Force redraw
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(1)

def stimulate_inferior_olive_cell(i_id=0):
    """Stimulate Inferior Olive"""
    stim = h.IClamp(inferior_olive_cells[i_id].soma(0.5))
    stim.delay = h.t
    stim.dur = 5
    stim.amp = 0.1
    stimuli.append(stim)

def update_inferior_olive_stimulation_and_plots(event=None, cell_nr=0):
    """Stimulates a inferior olive and updates the plots"""
    global buttons, animations
    
    if buttons["network_button"].label.get_text() == "Hide network":
        # Identify active purkinje cell
        b_id = 0
        if control == 0: # Control air pressure
            p_id_first = next((purkinje.gid for purkinje in purkinje_cells if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None)
        elif control == 1: # Control inflation time
            p_id_first = next((purkinje.gid for purkinje in purkinje_cells[:num_purkinje//2] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None)
            p_id_second = next((purkinje.gid for purkinje in purkinje_cells[num_purkinje//2:] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None)
        elif control == 2: # Control air pressure & inflation time
            p_id_first = next((purkinje.gid for purkinje in purkinje_cells[:num_purkinje//3] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None)
            p_id_second = next((purkinje.gid for purkinje in purkinje_cells[num_purkinje//3:2*num_purkinje//3] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None)
            p_id_third = next((purkinje.gid for purkinje in purkinje_cells[2*num_purkinje//3:] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None)
        elif control == 3: # Control grasp & hold
            p_id_first = next((purkinje.gid for purkinje in purkinje_cells[:num_purkinje//4] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None)
            p_id_second = next((purkinje.gid for purkinje in purkinje_cells[num_purkinje//4:2*num_purkinje//4] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None)
            p_id_third = next((purkinje.gid for purkinje in purkinje_cells[2*num_purkinje//4:3*num_purkinje//4] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None)
            p_id_fourth = next((purkinje.gid for purkinje in purkinje_cells[3*num_purkinje//4:] if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None)

        # Run complex spike animation
        spike, = ax_network.plot([], [], marker='o', color=color_complex_spike, markersize=10)
        spikes = [spike]
        i_ids = [cell_nr]
        p_ids = []
        
        if cell_nr == 0: # Trigger complex spike from IO 0
            p_ids.append(p_id_first)
        elif cell_nr == 1: # Trigger complex spike from IO 1
            p_ids.append(p_id_second)
        elif cell_nr == 2: # Trigger complex spike from IO 2
            p_ids.append(p_id_third)
        elif cell_nr == 3: # Trigger complex spike from IO 3
            p_ids.append(p_id_fourth)
        
        ani = animation.FuncAnimation(ax_network.figure, update_animation, frames=60, interval = 20, blit=True, repeat=False, fargs=(spikes, 1, p_ids, i_ids))
        animations.append(ani)
        plt.pause(5)
        #time.sleep(2)
        
    stimulate_inferior_olive_cell(i_id=cell_nr)
    
    update_spike_and_weight_plot()

def update_state(event=None):
    """Update state variable"""
    global state, buttons
    global pf_ncs, cf_ncs

    state = buttons["state_button"].index_selected if control == 3 else buttons["state_button"].index_selected + 1
    
    print(f"STATE: {state_grasp_hold_dict[state] if control == 3 else state_dict[state]}")

    if control == 3: # Control grasp & hold    
        if state == 0: # Only grasping
            try:
                buttons["error_holding"].disconnect_events()
                buttons["error_holding"].cla()
            except Exception: None
            # Success Button for Grasping
            ax_success = fig.add_subplot(gs_error[3], label="success_grasp")
            buttons["success_grasp"] = Button(ax_success, "Grasp\nSuccessful")
            buttons["success_grasp"].on_clicked(grasp_successfull)

            # Adapt GC->PC synapsis delay because PC gets more excitatory input because all GCs are active
            for purkinje in purkinje_cells:
                for granule in granule_cells:
                    pf_ncs[granule.gid][purkinje.gid].delay = 0.9

        else: # Grasp & Hold
            try:
                buttons["success_grasp"].disconnect_events()
                buttons["success_grasp"].cla()
            except Exception: None
            # Error Button for Holding Force
            ax_holding = fig.add_subplot(gs_error[3], label="error_holding")
            buttons["error_holding"] = Button(ax_holding, "Error\nHolding")
            buttons["error_holding"].on_clicked(lambda event: error_detected(event, btn_name="error_holding", cell_nr=3)) # stimulate IO cell 4

            # Reset GC->PC synapsis delay 
            for purkinje in purkinje_cells:
                for granule in granule_cells:
                    pf_ncs[granule.gid][purkinje.gid].delay = 0

    if buttons["network_button"].label.get_text() == "Hide network":
        ax_network.cla() # clear network plot
        ax_network.axis("off")
        show_network_graph()
    
    update_spike_and_weight_plot()

def toggle_HW(event=None):
    """Toggle between simulation of controlling HW"""
    global control_HW

    control_HW = next(i for i, value in hw_dict.items() if value == buttons["hardware_button"].value_selected)
 
    # Initialize HW
    if control_HW == 1:
        init_HW()
    else:
        release_actuator()

def grasp_successfull(event=None):
    """Clears all errors and changes colors of error buttons to default"""
    global errors, buttons

    try: # change color of "grasp successul" button
        buttons["success_grasp"].color = color_run
        buttons["success_grasp"].hovercolor = color_run_hover
        buttons["success_grasp"].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(1)

    errors = [False for _ in errors] # reset all errors
    change_back_error_button_colors()

def error_detected(event=None, btn_name=None, cell_nr=None):
    """Changes the color of the clicked error button and sets the correct error"""
    global buttons, errors

    if cell_nr != None and errors[cell_nr] == True: # if error is already set, reset error
        errors[cell_nr] = False
        color = "0.85"
        hovercolor = "0.975"
    else: # if error is not set, set error
        if cell_nr != None:
            errors[cell_nr] = True
        color = color_error
        hovercolor = color_error_hover

    try: # change color of clicked error button
        buttons[btn_name].color = color
        buttons[btn_name].hovercolor = hovercolor
        buttons[btn_name].ax.figure.canvas.draw_idle()  # Force redraw
    except KeyError: None
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.draw()
    plt.pause(1)
    time.sleep(0.1)
    
def toggle_control(event=None):
    """Toggle between controlling air pressure and inflation time"""
    global control, buttons, gs_error
    global num_inferior_olive, num_purkinje, num_dcn

    control = next(i for i, value in control_dict.items() if value == buttons["control_button"].value_selected)

    num_inferior_olive = control + 1
    num_dcn =  num_inferior_olive
    num_purkinje = 5 * (num_dcn)
    
    reset(event=None, reset_all=False)
    if control_HW==1:
        init_HW()

    # Change activation of error buttons
    if control == 0:
        try:
            buttons["error_button"].connect_events()
            gs_buttons.set_height_ratios([1,2,2,2.5,2.5,1,1,1,1])
        except Exception: None
    else:
        try:
            buttons["error_button"].disconnect_events()
            buttons["error_button"].ax.cla()
            buttons["error_button"].ax.axis("off")
            gs_error = gs_buttons[6].subgridspec(1, num_dcn)
            gs_buttons.set_height_ratios([1,2,2,2.5,2.5,1,2,1,1])
        except Exception: None
    
    # Change state labels
    if control == 3:
        buttons["state_button"].disconnect_events()
        buttons["state_button"].ax.cla()
        buttons["state_button"] = RadioButtons(buttons["state_button"].ax, list(state_grasp_hold_dict.values()), active=0)
        buttons["state_button"].on_clicked(update_state)
        #update_state()
    else:
        if len(buttons["state_button"].labels) == 4:
            buttons["state_button"].disconnect_events()
            buttons["state_button"].ax.cla()
            buttons["state_button"] = RadioButtons(buttons["state_button"].ax, list(state_dict.values()), active=1)
            buttons["state_button"].on_clicked(update_state)
            #update_state()

    if control == 1: # Control inflation time, one inferior_olive needed for each finger
        # Error Button for Thumb
        if "error_thumb" not in buttons:
            ax_thumb = fig.add_subplot(gs_error[0], label="error_thumb")
            buttons["error_thumb"] = Button(ax_thumb, "Error\nThumb")
            buttons["error_thumb"].on_clicked(lambda event: error_detected(event, btn_name="error_thumb", cell_nr=0)) # stimulate IO cell 0

        # Error Button for Index Finger
        if "error_index" not in buttons:
            ax_index = fig.add_subplot(gs_error[1], label="error_index")
            buttons["error_index"] = Button(ax_index, "Error\nIndex")
            buttons["error_index"].on_clicked(lambda event: error_detected(event, btn_name="error_index", cell_nr=1)) # stimulate IO cell 1
    
    elif control == 2: # Control air pressure & inflation time
        # Error Button for Air Pressure
        if "error_pressure" not in buttons:
            ax_pressure = fig.add_subplot(gs_error[0], label="error_pressure")
            buttons["error_pressure"] = Button(ax_pressure, "Error\nPressure")
            buttons["error_pressure"].on_clicked(lambda event: error_detected(event, btn_name="error_pressure", cell_nr=0)) # stimulate IO cell 2

        # Error Button for Thumb
        if "error_thumb" not in buttons:
            ax_thumb = fig.add_subplot(gs_error[1], label="error_thumb")
            buttons["error_thumb"] = Button(ax_thumb, "Error\nThumb")
            buttons["error_thumb"].on_clicked(lambda event: error_detected(event, btn_name="error_thumb", cell_nr=1)) # stimulate IO cell 0

        # Error Button for Index Finger
        if "error_index" not in buttons:
            ax_index = fig.add_subplot(gs_error[2], label="error_index")
            buttons["error_index"] = Button(ax_index, "Error\nIndex")
            buttons["error_index"].on_clicked(lambda event: error_detected(event, btn_name="error_index", cell_nr=2)) # stimulate IO cell 1

    elif control == 3: # Control grasp & hold
        # Error Button for Opposition
        if "error_opposition" not in buttons:
            ax_opposition = fig.add_subplot(gs_error[0], label="error_opposition")
            buttons["error_opposition"] = Button(ax_opposition, "Error\nOpposition\n&Extension")
            buttons["error_opposition"].on_clicked(lambda event: error_detected(event, btn_name="error_opposition", cell_nr=0)) # stimulate IO cell 0

        # Error Button for Index Finger Flexion
        if "error_index_flexion" not in buttons:
            ax_index = fig.add_subplot(gs_error[1], label="error_index_flexion")
            buttons["error_index_flexion"] = Button(ax_index, "Error\nIndex\nFlexion")
            buttons["error_index_flexion"].on_clicked(lambda event: error_detected(event, btn_name="error_index_flexion", cell_nr=1)) # stimulate IO cell 1

        # Error Button for Index Finger Extension
        if "error_index_extension" not in buttons:
            ax_index = fig.add_subplot(gs_error[2], label="error_index_extension")
            buttons["error_index_extension"] = Button(ax_index, "Error\nIndex\nExtension")
            buttons["error_index_extension"].on_clicked(lambda event: error_detected(event, btn_name="error_index_extension", cell_nr=2)) # stimulate IO cell 2

        if state == 0: # Only grasping
            if "success_grasp" not in buttons:
                ax_success = fig.add_subplot(gs_error[3], label="success_grasp")
                buttons["success_grasp"] = Button(ax_success, "Grasp\nSuccessful")
                buttons["success_grasp"].on_clicked(grasp_successfull)
        else: # Grasp & Hold
            if "error_holding" not in buttons:
                ax_holding = fig.add_subplot(gs_error[3], label="error_holding")
                buttons["error_holding"] = Button(ax_holding, "Error\nHolding")
                buttons["error_holding"].on_clicked(lambda event: error_detected(event, btn_name="error_holding", cell_nr=2)) # stimulate IO cell 3
    update_state()

def toggle_mode(event=None):
    """Toggle between manual and automatic feedback mode"""
    global serial_con, mode, sensor_references

    mode = next(i for i, value in mode_dict.items() if value == buttons["automatic_button"].value_selected)
    
    if mode == 1: # Trigger error automatically based on sensor feedback
        # Open the serial connection to ESP32 (adjust COMx port or /dev/ttyUSBx)
        if serial_con is None:
            serial_con = serial.Serial('COM11', 115200, timeout=1)  # Adjust port if needed                        

def toggle_network_graph(event=None):
    """Toggles between showing and hiding the network graph in the GUI"""
    global buttons, ax_network, gs
    if buttons["network_button"].label.get_text() == "Hide network":
        buttons["network_button"].label.set_text("Show network")
        ax_network.cla() # clear network plot
        ax_network.axis("off")
        gs.set_height_ratios([0.1, 1])
        gs_buttons.set_height_ratios([1,2,2,2.5,2.5,1,2,1,1])
    else:
        buttons["network_button"].label.set_text("Hide network")
        gs.set_height_ratios([0.9, 1])
        gs_buttons.set_height_ratios([2,2,2,2.5,2.5,1,2,1,1])
        show_network_graph()
        
    update_spike_and_weight_plot()

def draw_purkinje(ax, x, y, width=0.2, height=1, color='orange', line_width=0.01):
    """Draws a Purkinje neuron with dendrites and a separate soma."""
    purkinje_drawing = []

    # Dendritic tree
    for i in range(num_granule):  # Branching
        top_width =  (line_width[i] if np.isscalar(line_width) is not True else line_width)
        triangle = patches.Polygon([
            (x + (i-1) * (width if i > 0 else width/2) - top_width / 2, y + (i+1) * width),  # Left top
            (x + (i-1) * (width if i > 0 else width/2) + top_width / 2, y + (i+1) * width),  # Right top
            (x, y)  # Bottom center
        ], closed=True, color=color, alpha=0.6)  # Slight transparency

        ax.add_patch(triangle)
        purkinje_drawing.append(triangle)
    
    # Axons
    drawing = ax.plot([x, x], [y, y - height], color=color, lw=2)
    purkinje_drawing.append(drawing[0])

    # Soma (neuron body)
    drawing = ax.scatter(x, y, s=200, color=color, zorder=2)
    purkinje_drawing.append(drawing)

    return purkinje_drawing

def draw_parallel_fiber(ax, x, y, length=5, transparency=1):
    """Draws a parallel fiber extending across Purkinje cells."""
    ax.plot([x - length / 30, x + length], [y , y], color=color_granule, lw=2, alpha=transparency)

def draw_granule_to_parallel(ax, x, y_start, y_end, transparency):
    """Draws a granule cell axon that ascends vertically and forms a parallel fiber."""
    ax.plot([x, x], [y_start, y_end], color=color_granule, lw=2, alpha=transparency)  # Vertical axon
    draw_parallel_fiber(ax, x, y_end, transparency=transparency)  # Horizontal fiber

def draw_climbing_fiber(ax, x, y_start, y_end, width=0.3):
    """Draws a climbing fiber from the Inferior Olive wrapping around a Purkinje cell."""

    ax.plot([x + 0.15, x + 0.15], [y_start, y_end - 0.15], color=color_inferior_olive, lw=2, label="Climbing Fiber")
    ax.plot([x, x + 0.15], [y_end, y_end - 0.15], color=color_inferior_olive, lw=2, label="Climbing Fiber")

    for i in range(num_granule):  # Branching
        branch_x_start = x
        branch_x_end = x + (i-1) * (width if i > 0 else width/2)
        branch_y_start = y_end
        branch_y_end = y_end + (i+1) * width

        dx = branch_x_end - branch_x_start
        dy = branch_y_end - branch_y_start
        length = np.sqrt(dx**2 + dy**2)  
        if length == 0:
            continue  # Avoid division by zero

        t = np.linspace(0, length, 100)
        wave = 0.015 * np.sin(15 * np.pi * t)

        # Compute unit direction
        ux, uy = dx / length, dy / length  # Unit vector along branch
        nx, ny = -uy, ux  # Perpendicular vector for wave oscillation
        x_vals = branch_x_start + ux * t + wave * nx
        y_vals = branch_y_start + uy * t + wave * ny

        ax.plot(x_vals, y_vals, color=color_inferior_olive, lw=1, label="Climbing Fiber")

def calculate_dcn_x_positions(purkinje_x, num_dcn):
    """Calculated the x positions of Deep Cerebellar Nuclei"""
    dcn_x = []

    # Split the purkinje_x into equal segments and compute the middle points
    segment_length = len(purkinje_x) // num_dcn  # Calculate segment size
    
    for i in range(num_dcn):
        # Determine the segment range
        start_idx = i * segment_length
        if i == num_dcn - 1:
            # For the last segment, include all remaining elements
            end_idx = len(purkinje_x)
        else:
            end_idx = (i + 1) * segment_length
        
        # Calculate the average of the current segment
        segment_avg = np.mean(purkinje_x[start_idx:end_idx])
        dcn_x.append(segment_avg)

    return dcn_x

def update_weights_in_network():
    """Updated the weights in the network as increasing or decreasing triangles"""
    global ax_network, purkinje_drawing

    # --- Normalize Triangle Widths ---
    min_w, max_w = min(weights.values()), max(weights.values())
    triangle_widths = np.empty((num_granule, num_purkinje))

    if max_w > min_w:
        for g in range(num_granule):
            for p in range(num_purkinje):
                triangle_widths[g, p] = (weights[(g, p)] - min_w) / (max_w - min_w) * (max_w - min_w)
    else:
        triangle_widths.fill(0.5)

    # Update existing Purkinje triangles
    for i, purkinje_group in enumerate(purkinje_drawing):
        for j, triangle in enumerate(purkinje_group[:-2]):  # Ignore axon & soma
            if isinstance(triangle, patches.Polygon):  
                x, y = triangle.xy[2]  # Fixed bottom point
                top_width = triangle_widths[j, i]
                # Update triangle shape
                new_xy = [
                    (x + (j-1) * (width if j > 0 else width/2) - top_width / 2, y + (j+1) * width),  # Left top
                    (x + (j-1) * (width if j > 0 else width/2) + top_width / 2, y + (j+1) * width),  # Right top
                    (x, y)  # Bottom center
                ]
                triangle.set_xy(new_xy)  # Update vertices
    
    plt.draw()
    plt.pause(1)

def show_network_graph():
    """Shows the biologically inspired network graph of cerebellar cells and connections"""
    global ax_network, purkinje_drawing
    global height, width, granule_x, purkinje_x, olive_x, basket_x, dcn_x, granule_y, purkinje_y, olive_y, basket_y, dcn_y

    purkinje_drawing = []

    height = 1
    width = 0.2

    granule_x = np.linspace(0.15, 0.5, num_granule)
    purkinje_x = np.linspace(0.9, 4.8, num_purkinje)
    olive_x = purkinje_x[-1] + 0.4
    basket_x = purkinje_x[-1] + 0.4
    dcn_x = calculate_dcn_x_positions(purkinje_x, num_dcn)
    
    granule_y = -height*3/4 
    purkinje_y = 0
    olive_y = np.linspace(-0.9*height, -0.5*height, num_inferior_olive)
    basket_y = purkinje_y 
    dcn_y = -1.3*height

    # Draw Inferior Olive cell
    for inferior_olive in inferior_olive_cells:
        first_purkinje = next((purkinje for purkinje in purkinje_cells if cf_ncs[inferior_olive.gid][purkinje.gid] is not None), 0) # find first connected PC, default PC0
        ax_network.plot([purkinje_x[first_purkinje.gid]+0.15, olive_x], [olive_y[inferior_olive.gid], olive_y[inferior_olive.gid]], color=color_inferior_olive, lw=2, label="Climbing Fiber")
        ax_network.scatter(olive_x, olive_y[inferior_olive.gid], s=100, color=color_inferior_olive, label="Inferior Olive")

    # Draw Basket cell connecting to Purkinje cell somas
    ax_network.plot([purkinje_x[0], basket_x], [basket_y, basket_y], color=color_basket, lw=2)
    ax_network.scatter(basket_x, basket_y, s=100, color=color_basket, label="Basket Cell")
    
    # Draw Granule cells, vertical axons, and parallel fibers
    for granule in granule_cells:
        if state == 0 or granule.gid == state-1:
            transparency = 1
        else:
            transparency = 0.5
        ax_network.scatter(granule_x[granule.gid], granule_y, color=color_granule, s=100, label="Granule Cell", alpha=transparency) 
        draw_granule_to_parallel(ax_network, granule_x[granule.gid], granule_y, purkinje_y + (granule.gid+1) * width, transparency)

    # Draw Purkinje cells
    for purkinje in purkinje_cells:
        for inferior_olive in inferior_olive_cells: 
            if cf_ncs[inferior_olive.gid][purkinje.gid] is not None:  # Check if CF synapse exists
                draw_climbing_fiber(ax_network, purkinje_x[purkinje.gid], olive_y[inferior_olive.gid], purkinje_y, width=width)  # Climbing fibers
        drawing = draw_purkinje(ax_network, purkinje_x[purkinje.gid], purkinje_y, width=width, height=height, color=colors_purkinje[purkinje.gid])
        purkinje_drawing.append(drawing)

    # Draw Deep Cerebellar Nuclei
    for i in range(num_dcn):
        ax_network.scatter(dcn_x[i], dcn_y, color=color_dcn, s=100, label="Deep Cerebellar Nuclei") 
        ax_network.plot([dcn_x[i], dcn_x[i]], [purkinje_y-height, dcn_y], color=color_dcn, lw=2)

        segment_length = num_purkinje // num_dcn  # Length of each segment
    
        for i in range(num_dcn):
            # Determine the segment range
            start_idx = i * segment_length
            if i == num_dcn - 1:
                end_idx = len(purkinje_x)
            else:
                end_idx = (i + 1) * segment_length
            
            # Plot horizontal lines connecting Purkinje cells in each segment
            ax_network.plot([purkinje_x[start_idx], purkinje_x[end_idx-1]], [purkinje_y-height, purkinje_y-height], color=color_dcn, lw=2, label=f'Segment {i+1}')

    # Labels
    ax_network.text(purkinje_x[0] - 0.25, purkinje_y - 0.3, "Purkinje\nCells (PC)", fontsize=10, color=colors_purkinje[0])
    for purkinje in purkinje_cells:
        ax_network.text(purkinje_x[purkinje.gid] + 0.01, purkinje_y - 0.2, f"PC{purkinje.gid+1}", fontsize=10, color=colors_purkinje[purkinje.gid])
    ax_network.text(granule_x[0] - 0.05, granule_y - 0.4, "Granule Cells (GC)", fontsize=10, color=color_granule)
    for granule in granule_cells:
        ax_network.text(granule_x[granule.gid] - 0.05, granule_y - 0.2, f"GC{granule.gid+1}", fontsize=10, color=color_granule)
    ax_network.text(granule_x[1], purkinje_y + (num_granule) * width + 0.1, "Parallel Fibers (PF)", fontsize=10, color=color_granule)
    ax_network.text(olive_x + 0.1, olive_y[0] - 0.2, "Inferior Olives (IO)", fontsize=10, color=color_inferior_olive)
    for inferior_olive in inferior_olive_cells:
        ax_network.text(olive_x + 0.1, olive_y[inferior_olive.gid] - 0.04, f"IO{inferior_olive.gid+1}", fontsize=10, color=color_inferior_olive)
    ax_network.text(purkinje_x[-1] + 0.2, olive_y[-1] + abs(purkinje_y - olive_y[-1]) / 2, "Climbing Fibers (CF)", fontsize=10, color=color_inferior_olive)
    ax_network.text(basket_x + 0.1, basket_y - 0.01, "Basket Cell (BC)", fontsize=10, color=color_basket)
    ax_network.text(dcn_x[0] - 0.4, dcn_y, f"Deep Cerebellar\nNuclei (DCN)", fontsize=10, color=color_dcn)
    for i in range(num_dcn):
        if control == 0:
            text = DCN_names[control][i]
        elif control == 1:
            text = DCN_names[control][i]
        elif control == 2:
            text = DCN_names[control][1]
        elif control == 3:
            text = DCN_names[control][i]
        ax_network.text(dcn_x[i] + 0.05, dcn_y, f"DCN{i+1}\n'{text}'", fontsize=10, color=color_dcn)

    ax_network.set_xlim([0.0,6.0])

    update_weights_in_network()

    plt.draw()
    plt.pause(1)

def update_animation(frame, spikes, spike_type=0, p_ids=[], g_or_i_ids=[]):
    """Spike animation for simple spikes and complex spikes"""
    # Animation parameters
    total_steps = 60  # Total frames in animation
    
    for idx, spike in enumerate(spikes):
        # For each spike, get corresponding p_id and g_or_i_id
        p_id = p_ids[idx] if idx < len(p_ids) else 0  # Default to 0 if not enough IDs provided
        g_or_i_id = g_or_i_ids[idx] if idx < len(g_or_i_ids) else 0  # Default to 0 if not enough IDs
        
        # Determine start and end positions based on spike type
        if spike_type == 1:  # Complex Spike from Inferior Olive
            start_x, start_y = olive_x, olive_y[g_or_i_id]
            junction1_x, junction1_y = purkinje_x[p_id] + 0.15, start_y
            junction2_x, junction2_y = junction1_x, purkinje_y - 0.15
        else:  # Simple Spike from Granule Cell
            start_x, start_y = granule_x[g_or_i_id], granule_y
            junction1_x, junction1_y = start_x, purkinje_y + (g_or_i_id + 1) * width
            junction2_x, junction2_y = purkinje_x[p_id] + (g_or_i_id - 1) * (width if g_or_i_id > 0 else width/2), junction1_y
        end_x, end_y = purkinje_x[p_id], purkinje_y

        # Compute segment lengths
        d1 = np.hypot(junction1_x - start_x, junction1_y - start_y)
        d2 = np.hypot(junction2_x - junction1_x, junction2_y - junction1_y)
        d3 = np.hypot(end_x - junction2_x, end_y - junction2_y)
        D_total = d1 + d2 + d3

        # Allocate frames proportionally
        segment_steps1 = round(total_steps * (d1 / D_total))
        segment_steps2 = round(total_steps * (d2 / D_total))
        segment_steps3 = total_steps - (segment_steps1 + segment_steps2)

        # Determine current segment and compute clamped t
        if frame < segment_steps1:  # Move to Junction 1
            t = ((frame + 1) / segment_steps1)
            t = min(max(t, 0), 1)  # Ensure t is in [0,1]
            x_new = start_x + t * (junction1_x - start_x)
            y_new = start_y + t * (junction1_y - start_y)

        elif frame < segment_steps1 + segment_steps2:  # Move to Junction 2
            t = ((frame + 1 - segment_steps1) / segment_steps2)
            t = min(max(t, 0), 1)  # Ensure t is in [0,1]
            x_new = junction1_x + t * (junction2_x - junction1_x)
            y_new = junction1_y + t * (junction2_y - junction1_y)

        else:  # Move to Purkinje Cell
            t = ((frame + 1 - segment_steps1 - segment_steps2) / segment_steps3)
            t = min(max(t, 0), 1)  # Ensure t is in [0,1]
            x_new = junction2_x + t * (end_x - junction2_x)
            y_new = junction2_y + t * (end_y - junction2_y)

        # Update the spike's data
        spike.set_data([x_new], [y_new])

        # If last frame, hide the spike
        if frame == total_steps - 1:
            spike.set_alpha(0)

    return spikes  # Return all updated spikes

def reset(event=None, reset_all=True):
    """Resets the program"""
    global t, iter

    h.finitialize(-65)
    h.frecord_init()
    h.stdinit()
    h.t = 0

    if t is not None:
        t.resize(0)  # Clear old values
    else:
        t= h.Vector()

    init_variables(reset_all)
    create_connections()
    recording()
    
    run_simulation()
    iter += 1
    update_spike_and_weight_plot()

    buttons["run_button"].color = color_run
    buttons["run_button"].hovercolor = color_run_hover
    buttons["run_button"].ax.figure.canvas.draw_idle()  # Force redraw
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(1)

def update_weights(pre_gid, post_gid, pre_t, post_t):
    """STDP Update Function"""
    delta_t = post_t - pre_t # time between presynaptic spike and postsynaptic spike
    dw = 0
    plasticity = None
    if delta_t > 0: 
        dw = A_plus * np.exp(-delta_t / tau_plus)
        #if dw > 0.001: print(f"[{iter}] LTP: dw= {dw:.3f} GC{pre_gid+1} <-> PC{post_gid+1}")
        if dw > 0.001: 
            plasticity = "LTP"
    elif delta_t < 0:
        dw = -A_minus * np.exp(delta_t / tau_minus)
        #if dw < -0.001: print(f"[{iter}] LTD: dw={dw:.3f} GC{pre_gid+1} <-> PC{post_gid+1}")
        if dw < -0.001: 
            plasticity = "LTD"        

    old_weight = weights[(pre_gid, post_gid)]
    new_weight = old_weight + dw
    
    if plasticity:
        print(f"[{iter}] {plasticity}: weight change {old_weight:.4f}{'+' if dw>0 else ''}{dw:.4f}={new_weight:.4f} at synapses GC{pre_gid+1} <-> PC{post_gid+1}")
    
    # Update weights
    weights[(pre_gid, post_gid)] = new_weight
    pf_ncs[pre_gid][post_gid].weight[0] = new_weight
    
def recording():
    """Records Spiking Activity and Voltages"""
    global t, granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, v_granule, v_purkinje, v_inferiorOlive, v_basket

    t.record(h._ref_t)  # Reattach to NEURON's time

    granule_spikes = {i: h.Vector() for i in range(num_granule)}
    purkinje_spikes = {i: h.Vector() for i in range(num_purkinje)}
    inferiorOlive_spikes = {i: h.Vector() for i in range(num_inferior_olive)}
    basket_spikes = {i: h.Vector() for i in range(num_basket)}

    v_granule = {i: h.Vector().record(granule_cells[i].soma(0.5)._ref_v) for i in range(num_granule)}
    v_purkinje = {i: h.Vector().record(purkinje_cells[i].soma(0.5)._ref_v) for i in range(num_purkinje)}
    v_inferiorOlive = {i: h.Vector().record(inferior_olive_cells[i].soma(0.5)._ref_v) for i in range(num_inferior_olive)}
    v_basket = {i: h.Vector().record(basket_cells[i].soma(0.5)._ref_v) for i in range(num_basket)}

    for granule in granule_cells:
        nc = h.NetCon(granule.soma(0.5)._ref_v, None, sec=granule.soma)
        nc.record(granule_spikes[granule.gid])

    for purkinje in purkinje_cells:
        nc = h.NetCon(purkinje.soma(0.5)._ref_v, None, sec=purkinje.soma)
        nc.record(purkinje_spikes[purkinje.gid])

    for inferior_olive in inferior_olive_cells:
        nc = h.NetCon(inferior_olive.soma(0.5)._ref_v, None, sec=inferior_olive.soma)
        nc.record(inferiorOlive_spikes[inferior_olive.gid])

    for basket in basket_cells:
        nc = h.NetCon(basket.soma(0.5)._ref_v, None, sec=basket.soma)
        nc.record(basket_spikes[basket.gid])
    
    h.finitialize(-65) # Set all membrane potentials to -65mV
    
def run_simulation(error=False):
    """Runs the simulation for one iteration and tracks the weights"""
    global granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes
    global iter, spike_times, processed_GC_spikes, processed_pairs, frequency, weights_over_time
    global t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np

    try:
        buttons["run_button"].color = "0.85"
        buttons["run_button"].hovercolor = "0.975"
        buttons["run_button"].ax.figure.canvas.draw_idle()  # Force redraw
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(1)
    except Exception: None
    
    if error:
        time_span = 1/8 * 1/frequency*1000
        stop_time = h.t + time_span
    else:
        stop_time = 1/frequency*1000 * (iter + 1) # run 20 ms per iteration

    # Continuously run the simulation and update weights during the simulation
    while h.t < stop_time: 
        h.continuerun(h.t + 1)  # Incrementally run the simulation

        if not error:
            # --- Trigger Purkinje Cell Spike ---
            for g_id in range(num_granule):
                for pre_t in granule_spikes[g_id]:
                    if pre_t > stop_time - 1/frequency*1000: # timespan between last GC stimulation
                        if (pre_t) not in processed_GC_spikes[(g_id)]:
                            processed_GC_spikes[g_id].add((pre_t))

            # --- Apply STDP ---
            for g_id in range(num_granule):
                for p_id in range(num_purkinje):
                    for pre_t in granule_spikes[g_id]:
                        for post_t in purkinje_spikes[p_id]:
                            if (pre_t, post_t) not in processed_pairs[(g_id, p_id)]:
                                #print(f"update weights for GC{g_id+1} <-> PC{p_id+1} pre_t {pre_t:.2f} post_t {post_t:.2f}")
                                update_weights(g_id, p_id, pre_t, post_t)
                                processed_pairs[(g_id, p_id)].add((pre_t, post_t))

                    # Track the weight at the current time step
                    while len(weights_over_time[(g_id, p_id)]) < len(t):
                        weights_over_time[(g_id, p_id)].append(weights[(g_id, p_id)])
        else:
            for g_id in range(num_granule):
                for p_id in range(num_purkinje):
                    # Track the weight at the current time step
                    while len(weights_over_time[(g_id, p_id)]) < len(t):
                        weights_over_time[(g_id, p_id)].append(weights[(g_id, p_id)])
 
    # --- Convert Spike Data ---
    spike_times =      {f"GC{i+1}": list(granule_spikes[i])       for i in range(num_granule)}
    spike_times.update({f"PC{i+1}": list(purkinje_spikes[i])      for i in range(num_purkinje)})
    spike_times.update({f"IO{i+1}": list(inferiorOlive_spikes[i]) for i in range(num_inferior_olive)})
    spike_times.update({f"BC{i+1}": list(basket_spikes[i])        for i in range(num_basket)})
    
    # --- Convert Voltage Data and Weights ---
    t_np = np.array(t)
    v_granule_np =       np.array([vec.to_python() for vec in v_granule.values()])
    v_purkinje_np =      np.array([vec.to_python() for vec in v_purkinje.values()])
    v_inferiorOlive_np = np.array([vec.to_python() for vec in v_inferiorOlive.values()])
    v_basket_np =        np.array([vec.to_python() for vec in v_basket.values()])

def update_spike_and_weight_plot():
    """Updates the plots for the spikes and weights (one weight plot per DCN) and the buttons"""
    global t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np
    global buttons, fig, gs, ax_network, ax_plots, gs_buttons

    if gs == None or ax_network == None or ax_plots == None:
        try:
            gs = GridSpec(2, 1 + num_dcn + 1, figure=fig, width_ratios=(1+num_purkinje // (num_purkinje//num_dcn))*[1] + [0.35 * num_dcn], height_ratios=[0.1, 1])
        except AttributeError: None
        ax_network = fig.add_subplot(gs[0, :], label="ax_network")
        ax_network.axis("off")
        ax_plots = [None for _ in range(1 + num_dcn)]
        for col in range(1 + num_purkinje // (num_purkinje//num_dcn)):
            ax_plots[col] = fig.add_subplot(gs[1, col], label=f"ax_plots[{col}]")
        gs_buttons = gs[1, -1].subgridspec(9, 1, height_ratios=(1,2,2,2.5,2.5,1,1,1,1))
    else:
        # Clear previous plots
        for col in range(1 + num_dcn):
            ax_plots[col].cla()

    # Share axis
    for col in range(1 + num_dcn): 
        if col > 0:
            ax_plots[col].sharex(ax_plots[0])  # Share x-axis with first column
        if col > 1:
            ax_plots[col].sharey(ax_plots[1])  # Share y-axis with second column

    for granule in granule_cells:

        if granule.gid == (state-1) % num_granule:
            ax1 = ax_plots[0]
            ax1.set_title(f"Spiking Activity")
            ax1.plot(t_np, v_granule_np[granule.gid], label=f"GC{granule.gid+1 if state > 0 else ''}", color=color_granule, linestyle="dashed")
            ax1.set_xlabel("Time (ms)") 
            ax1.set_ylabel("Membrane Voltage (mV)")
    
            for i in range(num_dcn):
                ax2 = ax_plots[1 + i]
                if control == 0:
                    text = DCN_names[control][i]
                elif control == 1:
                    text = DCN_names[control][i]
                elif control == 2:
                    text = DCN_names[control][i]
                elif control == 3:
                    text = DCN_names[control][i]
                ax2.set_title(f"Synaptic Weights\n'{text}'")
                ax2.set_xlabel("Time (ms)")
                ax2.set_ylabel("Synaptic Weight")

                for purkinje in purkinje_cells:
                    text_blocked = ""
                    try:
                        if v_purkinje_np[purkinje.gid][-1] > -56:
                            text_blocked = " blocked"
                    except IndexError: None

                    # --- Spiking Plot for GC and its connected PCs ---
                    ax1.plot(t_np, v_purkinje_np[purkinje.gid], label=f"PC{purkinje.gid+1}{text_blocked}", color=colors_purkinje[purkinje.gid])

                    # --- Weight Plot for GC to all connected PCs ---
                    if len(weights_over_time[(granule.gid, purkinje.gid)]) > 0:
                        if purkinje.gid >= i * num_purkinje // num_dcn and purkinje.gid < (i + 1) * num_purkinje // num_dcn:
                            ax2.plot(t_np, weights_over_time[(granule.gid, purkinje.gid)], label=f"PC{purkinje.gid+1}{text_blocked}", color=colors_purkinje[purkinje.gid])

            for inferior_olive in inferior_olive_cells:
                ax1.plot(t_np, v_inferiorOlive_np[inferior_olive.gid], label=f"IO{inferior_olive.gid+1 if len(inferior_olive_cells) > 1 else ''}", color=color_inferior_olive, linestyle="dashed")
            for basket in basket_cells:
                ax1.plot(t_np, v_basket_np[basket.gid], label=f"BC{basket.gid+1 if len(basket_cells) > 1 else ''}", color=color_basket, linestyle="dashed")

    # Collect all legend handles and labels for the first column
    handles_first_row = []
    labels_first_row = []
    handles, labels = ax_plots[0].get_legend_handles_labels()
    for l, h in zip(labels, handles):
        if l not in labels_first_row:  # Avoid duplicates
            # Exclude Purkinje cells from the first legend
            if "PC" not in l:  # Only add non-Purkinje labels
                labels_first_row.append(l)
                handles_first_row.append(h)
    labels_first_row, handles_first_row = zip(*sorted(zip(labels_first_row, handles_first_row), key=lambda x: x[0]))
    
    # Create a single legend for the first column
    spacing = 0.1
    height = 0.5
    ncol_first_legend = 1
    ax_plots[0].legend(handles_first_row, labels_first_row, loc="upper left", bbox_to_anchor=(0, 1), ncol=ncol_first_legend, labelspacing=spacing, handleheight=height)
    legend_height_first_row = 1/20 * (len(labels_first_row) * (height + spacing) - spacing ) / ncol_first_legend
    while legend_height_first_row > ax_plots[0].get_position().height and ncol_first_legend < len(labels_first_row):
        ncol_first_legend += 1  # Increase the number of columns (max is the number of labels)
        ax_plots[0].legend(handles_first_row, labels_first_row, loc="upper left", bbox_to_anchor=(0, 1), ncol=ncol_first_legend, labelspacing=spacing, handleheight=height)
        legend_height_first_row = 1/20 * (len(labels_first_row) * (height + spacing) - spacing ) / ncol_first_legend
    
    # Collect all legend handles and labels for the other columns
    for col in range(1, num_inferior_olive + 1):
        handles_second_row = []
        labels_second_row = []
        handles, labels = ax_plots[col].get_legend_handles_labels()
        for l, h in zip(labels, handles):
            if l not in labels_second_row:
                labels_second_row.append(l)
                handles_second_row.append(h)
        
        # Create a single legend for the other columns
        ncol_second_legend = 1
        ax_plots[col].legend(handles_second_row, labels_second_row, loc="upper left", bbox_to_anchor=(0, 1), ncol=ncol_second_legend, labelspacing=spacing)
        legend_height_second_row = 1/20 * (len(labels_second_row) * (height + spacing) - spacing) / ncol_second_legend
        while legend_height_second_row > ax_plots[col].get_position().height and ncol_second_legend < len(labels_second_row):
            ncol_second_legend += 1  # Increase the number of columns (max is the number of labels)
            ax_plots[col].legend(handles_second_row, labels_second_row, loc="upper left", bbox_to_anchor=(0, 1), ncol=ncol_second_legend, labelspacing=spacing, handleheight=height)
            legend_height_second_row = 1/20 * (len(labels_second_row) * (height + spacing) - spacing) / ncol_second_legend

    # --- Buttons ---

    # Reset Button
    if "reset_button" not in buttons:
        reset_ax = fig.add_subplot(gs_buttons[8], label="reset_button")
        buttons["reset_button"] = Button(reset_ax, "Reset")
        buttons["reset_button"].on_clicked(reset)

    # Network Button
    if "network_button" not in buttons:
        network_ax = fig.add_subplot(gs_buttons[7], label="network_button")
        buttons["network_button"] = Button(network_ax, "Show network")
        buttons["network_button"].on_clicked(toggle_network_graph)

    # Error Button
    if "error_button" not in buttons:
        error_ax = fig.add_subplot(gs_buttons[6], label="error_button")
        buttons["error_button"] = Button(error_ax, "Error")
        buttons["error_button"].on_clicked(lambda event: update_inferior_olive_stimulation_and_plots(event, cell_nr=0))

    # Run Button
    if "run_button" not in buttons:
        run_ax = fig.add_subplot(gs_buttons[5], label="run_button")
        buttons["run_button"] = Button(run_ax, f"Run iteration {iter}")
        buttons["run_button"].on_clicked(update_granule_stimulation_and_plots)

    # State Button
    if "state_button" not in buttons:
        state_ax = fig.add_subplot(gs_buttons[4], label="state_button")
        buttons["state_button"] = RadioButtons(state_ax, list(state_dict.values()), active=state-1)
        buttons["state_button"].on_clicked(update_state)

    # Control Button
    if "control_button" not in buttons:
        control_ax = fig.add_subplot(gs_buttons[3], label="control_button")
        buttons["control_button"] = RadioButtons(control_ax, list(control_dict.values()), active=control)
        buttons["control_button"].on_clicked(toggle_control)

    # Hardware Button
    if "hardware_button" not in buttons:
        hardware_ax = fig.add_subplot(gs_buttons[2], label="hardware_button")
        buttons["hardware_button"] = RadioButtons(hardware_ax, list(hw_dict.values()), active=control_HW)
        buttons["hardware_button"].on_clicked(toggle_HW)

    # Automatic Button
    if "automatic_button" not in buttons:
        automatic_ax = fig.add_subplot(gs_buttons[1], label="automatic_button")
        buttons["automatic_button"] = RadioButtons(automatic_ax, list(mode_dict.values()), active=mode)
        buttons["automatic_button"].on_clicked(toggle_mode)

    plt.draw()
    plt.pause(1)

def main(reset=True):
    global t, granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes
    global t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np, iter

    init_variables(reset)
    create_connections()
    recording()

    #h.topology() # prints topology of network
    
    run_simulation()
    iter += 1
    update_spike_and_weight_plot()

    buttons["run_button"].color = color_run
    buttons["run_button"].hovercolor = color_run_hover
    buttons["run_button"].ax.figure.canvas.draw_idle()  # Force redraw
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(1)

    try:
        while True:
            plt.pause(10)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
        if control_HW == 1:
            release_actuator()
        plt.close()

    return

main()
    






