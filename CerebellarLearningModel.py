from neuron import h, gui
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.widgets import Button
from matplotlib.widgets import RadioButtons
from matplotlib.gridspec import GridSpec
import networkx as nx
import time

from pymata_aio.pymata3 import PyMata3
from pymata_aio.constants import Constants


# using Python 3.8.20

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

def init_variables():
    global iter, buttons, state, mode, control_HW, mode_dict, control_dict, environment, pc_voltage_mapping
    global fig, gs, ax_network, ax_plots, ax_buttons, animations, purkinje_drawing
    global num_granule, num_purkinje, num_inferior_olive, num_basket, granule_cells, purkinje_cells, inferior_olive_cells, basket_cells
    global pf_syns, pf_ncs, cf_syns, cf_ncs, inh_syns, inh_ncs
    global weights, weights_over_time, pf_initial_weight, cf_initial_weight, basket_initial_weight, max_weight, min_weight, stimuli, frequency, processed_GC_spikes, processed_pairs
    global tau_plus, tau_minus, A_plus, A_minus, dt_LTP, dt_LTD

    # --- GUI and Control ---
    iter = 0
    buttons = {}
    state = 1                           # 0: activated granule cell 0, 1: activated granule cell 1, 2: activated granule cell 2
    mode = 0                            # 0: manual, 1: automatic
    control_HW = 0                      # 0: simulation, 1: control HW
    mode_dict = {0:"Manual", 1:"Auto"}
    control_dict = {0:"Simulation", 1:"Control HW"}
    environment = {0 : 0, 1 : 2, 2 : 4} # "state : PC_ID" --> environment maps object_ID/state to the desired Purkinje Cell
    pc_voltage_mapping = {0 : 3.0,      # "PC_ID : actuator voltage" --> pc_voltage_mapping maps each Purkinje Cell to a specific voltage level to control the actuator
                          1 : 3.5, 
                          2 : 4.0, 
                          3 : 4.5, 
                          4 : 5.0}
    
    # --- Plotting ---
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(layout="constrained", figsize=[11,7])
    gs, ax_network, ax_plots, ax_buttons = None, None, None, None
    animations = []
    purkinje_drawing = []

    # --- Create Network ---
    num_granule = 3
    num_purkinje = 5
    num_inferior_olive = 1
    num_basket = 1

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
    pf_initial_weight = 0.01 # Parallel fiber initial weight
    cf_initial_weight = 0.5 # Climbing fiber initial weight
    basket_initial_weight = 0.1 # Basket to Purkinje weight
    max_weight = 0.1
    min_weight = 0.01
    stimuli = []
    frequency = 50 # Hz
    processed_GC_spikes = { (g_gid): set() for g_gid in range(num_granule)} # store the processed granule cell spikes
    processed_pairs = { (pre_id, post_id): set() for pre_id in range(num_granule) for post_id in range(num_purkinje) } # store the processed spike pairs for each (pre_id, post_id)

    # --- Learning Parameters ---
    tau_plus = 20  
    tau_minus = 20  
    A_plus = 0.005  
    A_minus = 0.005
    dt_LTP = 10  # Time window for LTP (ms)
    dt_LTD = -10  # Time window for LTD (ms)

def init_HW():
    global board, PushB1_pin, PushB2_pin, PushB3_pin, PushB4_pin, PushB5_pin, POT1_pin, POT2_pin, COMP_pin, PS1_pin, PS2_pin, Servo1_pin, Servo2_pin, Servo3_pin, Servo4_pin
    global Servo1_OUTLET, Servo1_INLET, Servo1_HOLD, Servo2_OUTLET, Servo2_INLET, Servo2_HOLD, Servo3_OUTLET, Servo3_INLET, Servo3_HOLD, Servo4_OUTLET, Servo4_INLET, Servo4_HOLD
    global PushB1_val_old, PushB2_val_old, PushB3_val_old, PushB4_val_old, PushB5_val_old
    #########################################
    # Upload StandardFirmata.ino to Arduino #
    #########################################
    
    # Open serial connection to Arduino
    windows_port = 'COM8'
    linux_port = '/dev/ttyUSB0'
    board = PyMata3(com_port=windows_port)

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
    Servo1_pin = 2 # connect to air tube 1 of Exoskeleton (Flexion)
    Servo2_pin = 3
    Servo3_pin = 4 # connect to air tube 2 of Exoskeleton (Extension)
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
    # Granule → Purkinje Connections (excitatory)
    for granule in granule_cells:
        for purkinje in purkinje_cells:
            syn = h.Exp2Syn(purkinje.soma(0.5))
            syn.e = 0 # Excitatory
            syn.tau1 = 1 # Synaptic rise time
            syn.tau2 = 5 # Synaptic decay time
            pf_syns[granule.gid][purkinje.gid] = syn
            nc = h.NetCon(granule.soma(0.5)._ref_v, syn, sec=granule.soma)
            nc.weight[0] = pf_initial_weight + np.random.uniform(0,0.001)
            nc.delay = 1
            pf_ncs[granule.gid][purkinje.gid] = nc
            weights[(granule.gid, purkinje.gid)] = nc.weight[0]
            

    # Inferior Olive → Purkinje Connections (excitatory)
    for inferior_olive in inferior_olive_cells:
        for purkinje in purkinje_cells:
            syn = h.Exp2Syn(purkinje.soma(0.5))
            syn.e = 0  # Excitatory
            syn.tau1 = 5 # Synaptic rise time
            syn.tau2 = 25 # Synaptic decay time
            cf_syns[inferior_olive.gid][purkinje.gid] = syn
            nc = h.NetCon(inferior_olive.soma(0.5)._ref_v, syn, sec=inferior_olive.soma)
            nc.weight[0] = cf_initial_weight
            nc.delay = 1
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
            nc.delay = 1
            inh_ncs[basket.gid][purkinje.gid] = nc

def activate_highest_weight_PC(granule_gid):
    global active_purkinje, inh_syns, inh_ncs, stimuli
    
    max_weight = -np.inf
    active_purkinje = None

    # Find the Purkinje cell with the highest weight
    for purkinje in purkinje_cells:
        try:
            #print(f"PC{purkinje.gid+1}, voltage: {v_purkinje_np[purkinje.gid][-1]} mV")
            if v_purkinje_np[purkinje.gid][-1] > -55: # if membrane voltage is above 50 mV
                #print(f"Skip PC{purkinje.gid+1}, voltage: {v_purkinje_np[purkinje.gid][-1]} mV")
                continue # Skip the blocked Purkinje cell
        except NameError: 
            None
        weight = pf_ncs[granule_gid][purkinje.gid].weight[0]
        if weight > max_weight:
            max_weight = weight
            active_purkinje = purkinje
    
    try:
        #print(f"Active purkinje: PC{active_purkinje.gid+1} with weight {max_weight}")
        None
    except NameError: 
        print("v_purkinje_np not defined")
    except AttributeError:
        print("All purkinje cells blocked")

    
    i_id = 0
    if active_purkinje != None:
        for purkinje in purkinje_cells:
            if purkinje == active_purkinje:
                # Activate connection to purkinje cell with highest weight
                #pf_ncs[granule_gid][purkinje.gid].weight[0] = pf_initial_weight
                #pf_ncs[granule_gid][purkinje.gid].weight[0] = weights[(granule_gid, purkinje.gid)]
                for basket in basket_cells:
                    inh_ncs[basket.gid][purkinje.gid].weight[0] = 0
                cf_ncs[i_id][purkinje.gid].weight[0] = cf_initial_weight
            else:
                for basket in basket_cells:
                    inh_ncs[basket.gid][purkinje.gid].weight[0] = basket_initial_weight
                
                # Deactivate connections to all other purkinje cells
                #pf_ncs[granule_gid][purkinje.gid].weight[0] = 0
                cf_ncs[i_id][purkinje.gid].weight[0] = 0
            
            #print(f"Granule {granule_gid+1} spiked at {spike_time} → Triggering Purkinje {active_purkinje.gid+1} (weight {max_weight})")
            #print(f"{h.t} PC{purkinje.gid+1} with weight {pf_ncs[granule_gid][purkinje.gid].weight[0]} and threshold {pf_ncs[granule_gid][purkinje.gid].threshold}")

def release_actuator():
    # Config servo pins
    board.servo_config(Servo1_pin)
    board.servo_config(Servo3_pin) 
    #board.sleep(1)
    # Set servos to outlet position to let air out
    board.analog_write(Servo1_pin, Servo1_OUTLET) # First release flexion
    board.sleep(1)
    board.analog_write(Servo3_pin, Servo3_OUTLET) # Then release extension
    board.sleep(1)
    # Reset servo pins
    board.set_pin_mode(Servo1_pin, Constants.INPUT)
    board.set_pin_mode(Servo3_pin, Constants.INPUT)

def control_actuator(voltage):
    board.servo_config(Servo1_pin) # Flexion 
    board.servo_config(Servo3_pin) # Extension
    board.analog_write(Servo1_pin, Servo1_INLET)
    board.analog_write(Servo3_pin, Servo3_INLET)
    board.sleep(2)
    if voltage is not None:
        board.analog_write(COMP_pin, int(voltage * 255 / 5))
    board.sleep(1.0)
    board.analog_write(Servo3_pin, Servo3_HOLD) # Stop extension
    board.sleep(1.0)
    board.analog_write(Servo1_pin, Servo1_HOLD) # Stop flexion  
    board.analog_write(COMP_pin, 0)
    # Reset servo pins
    board.set_pin_mode(Servo1_pin, Constants.INPUT)
    board.set_pin_mode(Servo3_pin, Constants.INPUT)

def stimulate_granule_cell():
    # --- Stimulate Granule Cells Based on State ---
    g_id = state
    
    stim = h.IClamp(granule_cells[g_id].soma(0.5))
    stim.delay = 1/frequency*1000 * (iter + 1/2)
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

def update_granule_stimulation_and_plots(event=None):
    global t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np, granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, buttons, iter, ax_network, animations
    g_id = state
    activate_highest_weight_PC(g_id)

    # Identify active purkinje cell
    b_id = 0
    p_id = next((purkinje.gid for purkinje in purkinje_cells if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None)
    
    if buttons["network_button"].label.get_text() == "Hide network":
        # Run simple spike animation
        spike, = ax_network.plot([], [], 'mo', markersize=15)
        ani = animation.FuncAnimation(ax_network.figure, update_animation, frames=30, interval = 50, blit=True, repeat=False, 
                                      fargs=(spike, 0, p_id, g_id))
        animations.append(ani)
        plt.pause(4)

    stimulate_granule_cell()
    [t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np] = run_simulation(granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes)
    iter += 1
    buttons["run_button"].label.set_text(f"Run iteration {iter}")

    if buttons["network_button"].label.get_text() == "Hide network":
        update_weights_in_network()

    update_spike_and_weight_plot(t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np)

    

    # --- Control of actuator ---
    if control_HW:
        actuator_voltage = pc_voltage_mapping[p_id] # look up table for purkinje cell to voltage mapping

        print("Releasing air...")
        release_actuator()
        board.sleep(1)
        print(f"PS1: {PS1_voltage:.2f}V, PS2: {PS2_voltage:.2f}V")

        
        print(f"Controlling actuator with {actuator_voltage}V (PC{p_id+1})")
        control_actuator(actuator_voltage)
        board.sleep(1)
        print(f"PS1: {PS1_voltage:.2f}V, PS2: {PS2_voltage:.2f}V")

# Stimulate Inferior Olive if previous activated PC resulted in an error
def stimulate_inferior_olive_cell():
    i_id = 0

    stim = h.IClamp(inferior_olive_cells[i_id].soma(0.5))
    stim.delay = h.t
    stim.dur = 5
    stim.amp = 0.1
    stimuli.append(stim)

    # Define complex spike burst + pause
    burst_start = h.t
    num_spikes = 3
    burst_freq = 10*frequency
    pause_duration = 5
    pause_current = -0.1

    burst_times = np.arange(burst_start, burst_start + (num_spikes * (1000 / burst_freq)), 1000 / burst_freq)
    time_points = list(burst_times) + [burst_times[-1] + 1, burst_times[-1] + 1 + pause_duration]
    current_values = [1.0] * len(burst_times) + [pause_current, 0]

    t_vec = h.Vector(time_points)
    i_vec = h.Vector(current_values)
    i_vec.play(stim._ref_amp, t_vec, True)

    # Send inhibitory signals to all purkinje cells expect active_purkinje
    #for basket in basket_cells:
    #    basket_stim = h.IClamp(basket.soma(0.5))
    #    basket_stim.delay = stim.delay -1 # same as inferior olive
    #    basket_stim.dur = stim.dur # same as inferior olive
    #    basket_stim.amp = stim.amp # same as inferior olive
    #    stimuli.append(basket_stim)

def update_inferior_olive_stimulation_and_plots(event=None):
    global t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np, granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, buttons, animations
    
    if buttons["network_button"].label.get_text() == "Hide network":
        # Identify active purkinje cell
        b_id = 0
        p_id = next((purkinje.gid for purkinje in purkinje_cells if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None)

        # Run complex spike animation
        spike, = ax_network.plot([], [], 'mo', markersize=15)
        ani = animation.FuncAnimation(ax_network.figure, update_animation, frames=30, interval = 50, blit=True, repeat=False, fargs=(spike, 1, p_id))
        animations.append(ani)
        plt.pause(4)

    stimulate_inferior_olive_cell()
    [t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np] = run_simulation(granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, error=True)
    
    if buttons["network_button"].label.get_text() == "Hide network":
        update_weights_in_network()
    
    update_spike_and_weight_plot(t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np) 

# Update state variable
def update_state(event):
    global state, buttons
    for i in range(3):
        if buttons["state_button"].value_selected == f"State {i+1}":
            state = i

    plt.draw()
    plt.pause(1)

# Toggle between simulation of controlling HW
def toggle_control(event=None):
    global control_HW

    control_HW = next(i for i, value in control_dict.items() if value == buttons["hardware_button"].value_selected)

    # Deactivate automatic mode when controlling HW
    buttons["automatic_button"].ax.set_visible(True if control_HW == 0 else False)
 
    # Initialize HW
    if control_HW == 1:
        init_HW()

    plt.draw()
    plt.pause(1)

# Toggle between manual and automatic mode
def toggle_mode(event=None):
    global state, mode, mode_dict, active_purkinje

    mode = next(i for i, value in mode_dict.items() if value == buttons["automatic_button"].value_selected)

    # Toggle button visibilities
    buttons["run_button"].ax.set_visible(True if mode == 0 else False)
    buttons["error_button"].ax.set_visible(True if mode == 0 else False)
    buttons["network_button"].ax.set_visible(True if mode == 0 else False)
    buttons["reset_button"].ax.set_visible(True if mode == 0 else False)

    # Trigger error automatically
    if mode == 1: # automatic mode
        for i in range(10):
            update_granule_stimulation_and_plots()
            if active_purkinje != None:
                if active_purkinje.gid != environment[state]:
                    print(f"PC{active_purkinje.gid+1} not desired, triggering error")
                    update_inferior_olive_stimulation_and_plots() # automatically trigger error
            if mode == 0:
                break

        buttons["automatic_button"].set_active(0)

def toggle_network_graph(event=None):
    global buttons, ax_network
    if buttons["network_button"].label.get_text() == "Hide network":
        buttons["network_button"].label.set_text("Show network")
        ax_network.cla() # clear network plot
        ax_network.axis("off")
        gs.set_height_ratios([0.1, 1, 1])
    else:
        buttons["network_button"].label.set_text("Hide network")
        show_network_graph()
        gs.set_height_ratios([1.5, 1, 1])
    
    plt.draw()
    plt.pause(1)

def draw_purkinje(ax, x, y, width=0.5, height=3, color='orange', line_width=2):
    """Draws a Purkinje neuron with dendrites and a separate soma."""
    purkinje_drawing = []

    # Dendritic tree
    for i in range(num_granule):  # Branching
        #drawing = ax.plot([x, x + (i-1) * width], [y, y + (i+1) * width], color=color, lw=line_width[i] if np.isscalar(line_width) is not True else line_width)
        #purkinje_drawing.append(drawing[0])
        top_width =  (line_width[i] if np.isscalar(line_width) is not True else line_width)
        triangle = patches.Polygon([
            (x + (i-1) * width - top_width / 2, y + (i+1) * width),  # Left top
            (x + (i-1) * width + top_width / 2, y + (i+1) * width),  # Right top
            (x, y)  # Bottom center
        ], closed=True, color=color, alpha=0.6)  # Slight transparency

        ax.add_patch(triangle)
        purkinje_drawing.append(triangle)
    
    # Axons
    drawing = ax.plot([x, x], [y, y - height], color=color, lw=4)
    purkinje_drawing.append(drawing[0])

    # Soma (neuron body)
    drawing = ax.scatter(x, y, s=200, color=color, zorder=2)
    purkinje_drawing.append(drawing)

    return purkinje_drawing

def draw_parallel_fiber(ax, x, y, length=5):
    """Draws a parallel fiber extending across Purkinje cells."""
    ax.plot([x - length / 10, x + length], [y , y], color='C9', lw=2)

def draw_granule_to_parallel(ax, x, y_start, y_end):
    """Draws a granule cell axon that ascends vertically and forms a parallel fiber."""
    ax.plot([x, x], [y_start, y_end], color='C9', lw=2)  # Vertical axon
    draw_parallel_fiber(ax, x, y_end)  # Horizontal fiber

def draw_climbing_fiber(ax, x, y_start, y_end, width=0.5):
    """Draws a climbing fiber from the Inferior Olive wrapping around a Purkinje cell."""

    ax.plot([x + 0.1, x + 0.1], [y_start, y_end - 0.1], color='black', lw=2, label="Climbing Fiber")
    ax.plot([x, x + 0.1], [y_end, y_end - 0.1], color='black', lw=2, label="Climbing Fiber")
    t = np.linspace(0, 1, 100)

    for i in range(num_granule):  # Branching
        branch_x_start = x
        branch_x_end = x + (i-1) * width
        branch_y_start = y_end
        branch_y_end = y_end + (i+1) * width
        x_vals = branch_x_start + (branch_x_end - branch_x_start) * t + 0.05 * np.sin(10 * np.pi * t)  # Wavy pattern for wrapping
        y_vals = branch_y_start + (branch_y_end - branch_y_start) * t + 0.05 * np.sin(10 * np.pi * t)
        ax.plot(x_vals, y_vals, color='black', lw=1, label="Climbing Fiber")

def update_weights_in_network():
    global ax_network, purkinje_drawing

    width = 0.5

    # --- Normalize Triangle Widths ---
    min_w, max_w = min_weight, max_weight
    triangle_widths = np.empty((num_granule, num_purkinje))

    if max_w > min_w:
        for g in range(num_granule):
            for p in range(num_purkinje):
                triangle_widths[g, p] = (weights[(g, p)] - min_w) / (max_w - min_w) / 4
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
                    (x + (j-1) * width - top_width / 2, y + (j+1) * width),  # Left top
                    (x + (j-1) * width + top_width / 2, y + (j+1) * width),  # Right top
                    (x, y)  # Bottom center
                ]
                triangle.set_xy(new_xy)  # Update vertices
    
    plt.draw()
    plt.pause(1)

def show_network_graph():
    global ax_network, purkinje_drawing

    height = 1
    width = 0.5

    purkinje_x = np.linspace(-1, 2, num_purkinje)
    granule_x = np.linspace(-2.5, -2, num_granule)
    olive_x = purkinje_x[-1] + 0.4
    basket_x = purkinje_x[-1] + 0.4
    
    purkinje_y = 0
    granule_y = -height*3/4  # Bottom row
    basket_y = purkinje_y
    olive_y = -height*3/4  # Inferior Olive position

    purkinje_colors = ["C0", "C1", "C2", "C3", "C4"]

    # --- Define and Normalize Line Widths ---
    line_widths = np.empty((num_granule, num_purkinje))
    #min_w, max_w = min(weights.values()), max(weights.values())
    min_w, max_w = min_weight, max_weight
    if max_w > min_w:  # Avoid division by zero
        for g in range(num_granule):
            for p in range(num_purkinje):
                line_widths[g,p] = (weights[(g,p)] - min_w) / (max_w - min_w)
    else:
        for g in range(num_granule):
            for p in range(num_purkinje):
                line_widths[g,p] = 2  # Default width if all weights are the same


    # Draw Inferior Olive cell
    ax_network.plot([purkinje_x[0]+0.1, olive_x], [olive_y, olive_y], color='black', lw=2, label="Climbing Fiber")
    ax_network.scatter(olive_x, olive_y, s=200, color='black', label="Inferior Olive")

    # Draw Basket cell connecting to Purkinje cell somas
    ax_network.plot([purkinje_x[0], basket_x], [basket_y, basket_y], color='C8', lw=2)
    ax_network.scatter(basket_x, basket_y, s=150, color='C8', label="Basket Cell")
    
    # Draw Granule cells, vertical axons, and parallel fibers
    for i, x in enumerate(granule_x):
        ax_network.scatter(x, granule_y, color='C9', s=100, label="Granule Cell") 
        draw_granule_to_parallel(ax_network, x, granule_y, purkinje_y + (i+1) * width)

    # Draw Purkinje cells
    for i, x in enumerate(purkinje_x):
        draw_climbing_fiber(ax_network, x, olive_y, purkinje_y, width=width)  # Climbing fibers
        drawing = draw_purkinje(ax_network, x, purkinje_y, width=width, height=height, color=purkinje_colors[i], line_width=line_widths[:, i])
        purkinje_drawing.append(drawing)

    # Labels
    ax_network.text(purkinje_x[0] - 0.7, purkinje_y - 0.4, "Purkinje Cells", fontsize=12, color=purkinje_colors[0])
    for i, x in enumerate(purkinje_x):
        ax_network.text(purkinje_x[i] - 0.3, purkinje_y - 0.2, f"PC{i+1}", fontsize=12, color=purkinje_colors[i])
    ax_network.text(granule_x[0] - 0.1, granule_y - 0.4, "Granule Cells", fontsize=12, color="C9")
    for i, x in enumerate(granule_x):
        ax_network.text(granule_x[i] - 0.1, granule_y - 0.2, f"GC{i+1}", fontsize=12, color="C9")
    ax_network.text(granule_x[1], purkinje_y + (num_granule) * width + 0.1, "Parallel Fibers (PF)", fontsize=12, color="C9")
    ax_network.text(olive_x + 0.2, olive_y, "Inferior Olive (IO)", fontsize=12, color="black")
    ax_network.text(purkinje_x[-1] + 0.2, olive_y + abs(purkinje_y - olive_y) / 2, "Climbing Fibers (CF)", fontsize=12, color="black")
    ax_network.text(basket_x + 0.2, basket_y, "Basket Cell (BC)", fontsize=12, color="C8")

    plt.draw()
    plt.pause(1)

def update_animation(frame, spike, spike_type=0, p_id=0, g_id=0): # spike_type = 0 (simple) or 1 (complex)

    # Animation parameters
    total_steps = 30  # Total frames in animation
    segment_steps = total_steps // 3  # Frames per segment
    
    # Define cell positions
    height = 1
    width = 0.5
    purkinje_x = np.linspace(-1, 2, num_purkinje)
    granule_x = np.linspace(-2.5, -2, num_granule)
    olive_x = purkinje_x[-1] + 0.4
    purkinje_y = 0
    granule_y = -height 
    olive_y = -height*3/4  
    
    if spike_type == 1: # Complex Spike from Inferior Olive
        start_x, start_y = olive_x, olive_y
        junction1_x, junction1_y = purkinje_x[p_id] + 0.1, start_y
        junction2_x, junction2_y = junction1_x, purkinje_y - 0.1
    else: # Simple Spike from Granule Cell
        start_x, start_y = granule_x[g_id], granule_y
        junction1_x, junction1_y = start_x, purkinje_y + (g_id+1) * width
        junction2_x, junction2_y = purkinje_x[p_id] + (g_id-1) * width, junction1_y
    end_x, end_y = purkinje_x[p_id], purkinje_y

    # Determine current segment and compute clamped t
    if frame < segment_steps:  # Move to Junction 1
        t = ((frame+1) / segment_steps)
        t = min(max(t, 0), 1)  # Ensure t is in [0,1]
        x_new = start_x + t * (junction1_x - start_x)
        y_new = start_y + t * (junction1_y - start_y)

    elif frame < 2 * segment_steps:  # Move to Junction 2
        t = ((frame + 1 - segment_steps) / segment_steps)
        t = min(max(t, 0), 1)  # Ensure t is in [0,1]
        x_new = junction1_x + t * (junction2_x - junction1_x)
        y_new = junction1_y + t * (junction2_y - junction1_y)

    else:  # Move to Purkinje Cell
        t = ((frame + 1 - 2 * segment_steps) / segment_steps)
        t = min(max(t, 0), 1)  # Ensure t is in [0,1]
        x_new = junction2_x + t * (end_x - junction2_x)
        y_new = junction2_y + t * (end_y - junction2_y)

    spike.set_data([x_new], [y_new])

    if frame == total_steps - 1: # Last frame --> hide the spike
        spike.set_alpha(0)

    return [spike]
    
def reset(event):
    None

# --- STDP Update Function ---
def update_weights(pre_gid, post_gid, delta_t, t):
    if delta_t > 0 and delta_t < dt_LTP:  
        dw = A_plus * np.exp(-delta_t / tau_plus)
        print(f"[{iter}] LTP: GC{pre_gid+1} <-> PC{post_gid+1}")
    elif delta_t < 0 and delta_t > dt_LTD:  
        dw = -A_minus * np.exp(delta_t / tau_minus)
        print(f"[{iter}] LTD: GC{pre_gid+1} <-> PC{post_gid+1}")
    else: dw = 0
    new_weight = weights[(pre_gid, post_gid)] + dw
    weights[(pre_gid, post_gid)] = np.clip(new_weight, min_weight, max_weight)
    # Update netcon weight
    pf_ncs[pre_gid][post_gid].weight[0] = np.clip(new_weight, min_weight, max_weight)
    
def recording():
    # --- Record Spiking Activity and Voltages---
    t = h.Vector().record(h._ref_t)
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
    
  

    return [t, granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, v_granule, v_purkinje, v_inferiorOlive, v_basket]

def run_simulation(granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, error = False):
    global iter, spike_times, processed_GC_spikes, processed_pairs, frequency

    if error:
        time_span = 1/4 * 1/frequency*1000
        stop_time = h.t + time_span
    else:
        stop_time = 1/frequency*1000 * (iter + 1) # run 20 ms per iteration


    # Continuously run the simulation and update weights during the simulation
    while h.t < stop_time: 
        h.continuerun(h.t + 1)  # Incrementally run the simulation
        
        if error == False:
            # --- Trigger Purkinje Cell Spike ---
            for g_id in range(num_granule):
                for pre_t in granule_spikes[g_id]:
                    #if pre_t > h.t -1:
                    if pre_t > stop_time - 1/frequency*1000: # timespan between last GC stimulation
                        if (pre_t) not in processed_GC_spikes[(g_id)]:
                            #print(f"{h.t} Stimulate highest weight PC for GC{g_id+1} with spike time {pre_t}")
                            #activate_highest_weight_PC(g_id)
                            processed_GC_spikes[g_id].add((pre_t))

            # --- Apply STDP ---
            for g_id in range(num_granule):
                for p_id in range(num_purkinje):
                    for pre_t in granule_spikes[g_id]:
                        for post_t in purkinje_spikes[p_id]:
                            if (pre_t, post_t) not in processed_pairs[(g_id, p_id)]:
                                delta_t = post_t - pre_t
                                #print(f"update weights for GC{g_id+1} <-> PC{p_id+1} pre_t {pre_t} post_t {post_t}")
                                update_weights(g_id, p_id, delta_t, h.t)
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

    return [t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np]

def update_spike_and_weight_plot(t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np):
    global buttons, fig, gs, ax_network, ax_plots, ax_buttons
    
    if gs == None or ax_network == None or ax_plots == None or ax_buttons == None:
        gs = GridSpec(3, num_granule + 1, figure = fig, width_ratios=[1,1,1,0.3], height_ratios=[0.1,1,1])
        ax_network = fig.add_subplot(gs[0, :])
        ax_network.axis("off")
        ax_plots = [[None for _ in range(num_granule + 1)] for _ in range(1,3)]
        for row in range (1,3): # reserve first row for network graph
            for col in range(num_granule):
                ax_plots[row-1][col] = fig.add_subplot(gs[row, col])
        ax_buttons = fig.add_subplot(gs[1:, -1])
        ax_buttons.axis("off")
        #ax_plots = fig.add_subplot(gs[1:, :]) 
            #fig, axes = plt.subplots(3, num_granule, figsize=(3 * num_granule, 8), gridspec_kw={'height_ratios': [1] + [3] * 2})
    else:
        # Clear previous plots
        #ax_plots = axes[1:, :] # reserve first row / upper part for network graph
        for row in range(2):
            for col in range(num_granule):
                ax_plots[row][col].cla()

    # Share y-axis within each row
    for row in range(2):
        for col in range(num_granule):  # Start from second column
            if row > 0: 
                ax_plots[row][col].sharex(ax_plots[0][col])  # Share x-axis with first row
            if col > 0:
                ax_plots[row][col].sharey(ax_plots[row][0])  # Share y-axis with first column

    io_id = 0
    b_id = 0
    for granule in granule_cells:
        
        ax1 = ax_plots[0][granule.gid]
        ax1.set_title(f"GC{granule.gid+1} Spiking Activity")
        ax1.plot(t_np, v_granule_np[granule.gid], label=f"GC{granule.gid+1}", color="C9")
        ax1.plot(t_np, v_inferiorOlive_np[io_id], label=f"IO", color="black")
        ax1.plot(t_np, v_basket_np[b_id], label=f"BC", color="C8")

        ax2 = ax_plots[1][granule.gid]
        ax2.set_title(f"GC{granule.gid+1} Synaptic Weights")
        ax2.set_xlabel("Time (ms)")    

        for purkinje in purkinje_cells:
            text_blocked = ""
            if v_purkinje_np[purkinje.gid][-1] > -55:
                text_blocked = " blocked"


            # --- Spiking Plot for GC and its connected PCs ---
            ax1.plot(t_np, v_purkinje_np[purkinje.gid], label=f"PC{purkinje.gid+1}{text_blocked}", linestyle="dashed")

            # --- Weight Plot for GC to all connected PCs ---
            if len(weights_over_time[(granule.gid, purkinje.gid)]) > 0:
                ax2.plot(t_np, weights_over_time[(granule.gid, purkinje.gid)], label=f"PC{purkinje.gid+1}{text_blocked}")
            
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper left')

    # Label y-axes only on the first column
    ax_plots[0][0].set_ylabel("Membrane Voltage (mV)")
    ax_plots[1][0].set_ylabel("Synaptic Weight")

    # --- Buttons ---
    # Automatic Button
    if "automatic_button" not in buttons:
        automatic_ax = fig.add_axes([0.9, 0.45, 0.1, 0.1])
        buttons["automatic_button"] = RadioButtons(automatic_ax, (mode_dict[0], mode_dict[1]), active=mode)
        buttons["automatic_button"].on_clicked(toggle_mode)

    # Hardware Button
    if "hardware_button" not in buttons:
        hardware_ax = fig.add_axes([0.9, 0.35, 0.1, 0.1])
        buttons["hardware_button"] = RadioButtons(hardware_ax, (control_dict[0], control_dict[1]), active=control_HW)
        buttons["hardware_button"].on_clicked(toggle_control)

    # State Button
    if "state_button" not in buttons:
        state_ax = fig.add_axes([0.9, 0.25, 0.1, 0.1])
        buttons["state_button"] = RadioButtons(state_ax, ('State 1', 'State 2', 'State 3'), active=state)
        buttons["state_button"].on_clicked(update_state)

    # Run Button
    if "run_button" not in buttons:
        run_ax = fig.add_axes([0.9, 0.20, 0.1, 0.05])
        buttons["run_button"] = Button(run_ax, f"Run iteration {iter}")
        buttons["run_button"].on_clicked(update_granule_stimulation_and_plots)

    # Error Button
    if "error_button" not in buttons:
        error_ax = fig.add_axes([0.9, 0.15, 0.1, 0.05])
        buttons["error_button"] = Button(error_ax, "Error")
        buttons["error_button"].on_clicked(update_inferior_olive_stimulation_and_plots)

    # Network Button
    if "network_button" not in buttons:
        network_ax = fig.add_axes([0.9, 0.10, 0.1, 0.05])
        buttons["network_button"] = Button(network_ax, "Show network")
        buttons["network_button"].on_clicked(toggle_network_graph)

    # Reset Button
    if "reset_button" not in buttons:
        reset_ax = fig.add_axes([0.9, 0.05, 0.1, 0.05])
        buttons["reset_button"] = Button(reset_ax, "Reset")
        buttons["reset_button"].on_clicked(reset)

    plt.draw()
    plt.pause(1)

def main():
    global t, granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, v_granule, v_purkinje, v_inferiorOlive, v_basket, t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np, fig, axes, iter
    init_variables()
    create_connections()
    [t, granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, v_granule, v_purkinje, v_inferiorOlive, v_basket] = recording()
    h.finitialize(-65)
    [t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np] = run_simulation(granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes)
    iter += 1
    update_spike_and_weight_plot(t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np)
    

main()



try:
    while True:
        time.sleep(0.1) # Delay between iterations

        # Update the plot
        plt.draw()
        plt.pause(1)


        if control_HW == 1:
            try:
                # Pressure sensor
                PS1_val = board.analog_read(PS1_pin)
                PS2_val = board.analog_read(PS2_pin)
                PS1_voltage = PS1_val * 5 / 1023
                PS2_voltage = PS2_val * 5 / 1023

            except NameError:
                None

except KeyboardInterrupt:
    print("Simulation stopped by user.")
    if control_HW == 1:
        release_actuator()
    plt.show()






