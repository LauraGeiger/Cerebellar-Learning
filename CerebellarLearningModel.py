from neuron import h, gui
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from matplotlib.widgets import RadioButtons
from matplotlib.gridspec import GridSpec
import networkx as nx
import time






# using Python 3.8.20

# --- Variable Initialization for plotting
plt.ion()  # Turn on interactive mode
fig = plt.figure(layout="constrained")
gs, ax_network, ax_plots, ax_buttons = None, None, None, None
#axes = None

# --- Granule, Purkinje, and Inferior Olive Cell Classes ---
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

# --- Create Network ---
num_granule = 3
num_purkinje = 5
num_inferior_olive = 1
num_basket = 1

granule_cells = [GranuleCell(i) for i in range(num_granule)]
purkinje_cells = [PurkinjeCell(i) for i in range(num_purkinje)]
inferior_olive_cells = [InferiorOliveCell(i) for i in range(num_inferior_olive)]
basket_cells = [BasketCell(i) for i in range(num_basket)]

# --- STDP Parameters ---
tau_plus = 20  
tau_minus = 20  
A_plus = 0.005  
A_minus = 0.005
dt_LTP = 10  # Time window for LTP (ms)
dt_LTD = -10  # Time window for LTD (ms)
pf_initial_weight = 0.01 # Parallel fiber initial weight
cf_initial_weight = 0.5 # Climbing fiber initial weight
basket_initial_weight = 0.1 # Basket to Purkinje weight
max_weight = 0.1
min_weight = 0.001


# --- Create Synapses and Connections ---
pf_syns = [[None for _ in range(num_purkinje)] for _ in range(num_granule)] # parallel fiber synapses
pf_ncs = [[None for _ in range(num_purkinje)] for _ in range(num_granule)] # parallel fiber netcons
cf_syns = [[None for _ in range(num_purkinje)] for _ in range(num_inferior_olive)] # climbing fiber synapses
cf_ncs = [[None for _ in range(num_purkinje)] for _ in range(num_inferior_olive)] # climbing fiber netcons
inh_syns = [[None for _ in range(num_purkinje)] for _ in range(num_basket)] # inhibitory synapses
inh_ncs = [[None for _ in range(num_purkinje)] for _ in range(num_basket)] # inhibitory netcons

stimuli = []


def init_variables():
    global iter, state, mode, mode_dict, environment, animations, frequency, weights, weights_over_time, processed_GC_spikes, processed_pairs, buttons, purkinje_drawing

    iter = 0
    state = 1  # User can change this state (0, 1, or 2) based on desired behavior
    mode = 0
    mode_dict = {0:"Manual", 1:"Auto"}
    environment = {0:0, 1:2, 2:4} # "state:PC_ID" environment maps object_ID/state to the desired Purkinje Cell
    animations = []
    frequency = 50 # Hz
    weights = {}
    weights_over_time = { (pre_gid, post_gid): [] for pre_gid in range(num_granule) for post_gid in range(num_purkinje) } # track weights over time
    processed_GC_spikes = { (g_gid): set() for g_gid in range(num_granule)} # store the processed granule cell spikes
    processed_pairs = { (pre_id, post_id): set() for pre_id in range(num_granule) for post_id in range(num_purkinje) } # store the processed spike pairs for each (pre_id, post_id)
    buttons = {}
    purkinje_drawing = []


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

    if buttons["network_button"].label.get_text() == "Hide network":
        b_id = 0
        p_id = next((purkinje.gid for purkinje in purkinje_cells if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None)
        spike, = ax_network.plot([], [], 'mo', markersize=15)
        ani = animation.FuncAnimation(ax_network.figure, update_animation, frames=30, interval = 50, blit=True, repeat=False, 
                                      fargs=(spike, 0, p_id, g_id))
        animations.append(ani)

    stimulate_granule_cell()
    [t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np] = run_simulation(granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes)
    iter += 1
    buttons["run_button"].label.set_text(f"Run iteration {iter}")
    update_spike_and_weight_plot(t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np)

    if buttons["network_button"].label.get_text() == "Hide network":
        update_weights_in_network()


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
        b_id = 0
        p_id = next((purkinje.gid for purkinje in purkinje_cells if inh_ncs[b_id][purkinje.gid].weight[0] == 0), None)

        spike, = ax_network.plot([], [], 'mo', markersize=15)
        ani = animation.FuncAnimation(ax_network.figure, update_animation, frames=30, interval = 50, blit=True, repeat=False, fargs=(spike, 1, p_id))
        animations.append(ani)

    stimulate_inferior_olive_cell()
    [t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np] = run_simulation(granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, error=True)
    update_spike_and_weight_plot(t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np)
    
    if buttons["network_button"].label.get_text() == "Hide network":
        update_weights_in_network()


# Update state variable
def update_state(event):
    global state, buttons
    for i in range(3):
        if buttons["state_button"].value_selected == f"State {i+1}":
            state = i

def toggle_mode(event=None):
    global state, mode, mode_dict, active_purkinje

    mode = next(i for i, value in mode_dict.items() if value == buttons["automatic_button"].value_selected)

    # Toggle button visibilities
    #buttons["state_button"].ax.set_visible(True if mode == 0 else False)
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

def toggle_network_graph(event):
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

def draw_purkinje(ax, x, y, width=0.5, height=3, color='orange', line_width=[2]*num_granule):
    """Draws a Purkinje neuron with dendrites and a separate soma."""
    purkinje_drawing = []

    # Dendritic tree
    for i in range(num_granule):  # Branching
        drawing = ax.plot([x, x + (i-1) * width], [y, y + (i+1) * width], color=color, lw=line_width[i])
        purkinje_drawing.append(drawing[0])
    
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

    height = 1
    width = 0.5

    purkinje_x = np.linspace(-1, 2, num_purkinje)
    purkinje_y = 0

    purkinje_colors = ["C0", "C1", "C2", "C3", "C4"]

    # --- Define and Normalize Line Widths ---
    line_widths = np.empty((num_granule, num_purkinje))
    min_w, max_w = min_weight, max_weight
    if max_w > min_w:  # Avoid division by zero
        for g in range(num_granule):
            for p in range(num_purkinje):
                line_widths[g,p] = (weights[(g,p)] - min_w) / (max_w - min_w) * 5 + 1  # Scale to range 1-6
    else:
        for g in range(num_granule):
            for p in range(num_purkinje):
                line_widths[g,p] = 2  # Default width if all weights are the same

    # Remove Purkinje cells
    for purkinje in purkinje_drawing:
        for drawing in purkinje:
            drawing.remove()
    purkinje_drawing = []
    # Draw Purkinje cells with updated weights
    for i, x in enumerate(purkinje_x):
        drawing = draw_purkinje(ax_network, x, purkinje_y, width=width, height=height, color=purkinje_colors[i], line_width=line_widths[:, i])
        purkinje_drawing.append(drawing)

def show_network_graph():
    global ax_network, purkinje_drawing
    #ax = fig.add_subplot(gs[0, :]) # reserve first row for network graph

    height = 1
    width = 0.5

    purkinje_x = np.linspace(-1, 2, num_purkinje)
    granule_x = np.linspace(-2.5, -2, num_granule)
    olive_x = purkinje_x[-1] + 0.4
    basket_x = purkinje_x[-1] + 0.4
    
    purkinje_y = 0
    granule_y = -height  # Bottom row
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
                line_widths[g,p] = (weights[(g,p)] - min_w) / (max_w - min_w) * 5 + 1  # Scale to range 1-6
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
    ax_network.text(purkinje_x[0] - 0.7, purkinje_y - 0.7, "Purkinje Cells", fontsize=12, color=purkinje_colors[0])
    for i, x in enumerate(purkinje_x):
        ax_network.text(purkinje_x[i] - 0.3, purkinje_y - 0.5, f"PC{i+1}", fontsize=12, color=purkinje_colors[i])
    ax_network.text(granule_x[0] - 0.1, granule_y - 0.7, "Granule Cells", fontsize=12, color="C9")
    for i, x in enumerate(granule_x):
        ax_network.text(granule_x[i] - 0.1, granule_y - 0.4, f"GC{i+1}", fontsize=12, color="C9")
    ax_network.text(granule_x[1], purkinje_y + (num_granule) * width + 0.2, "Parallel Fibers (PF)", fontsize=12, color="C9")
    ax_network.text(olive_x + 0.2, olive_y, "Inferior Olive (IO)", fontsize=12, color="black")
    ax_network.text(purkinje_x[-1] + 0.2, olive_y + abs(purkinje_y - olive_y) / 2, "Climbing Fibers (CF)", fontsize=12, color="black")
    ax_network.text(basket_x + 0.2, basket_y, "Basket Cell (BC)", fontsize=12, color="C8")

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


    # --- Button ---

    # Automatic Button
    if "automatic_button" not in buttons:
        automatic_ax = fig.add_axes([0.9, 0.5, 0.07, 0.1])
        buttons["automatic_button"] = RadioButtons(automatic_ax, (mode_dict[0], mode_dict[1]), active=mode)
        buttons["automatic_button"].on_clicked(toggle_mode)

    # State Button
    if "state_button" not in buttons:
        state_ax = fig.add_axes([0.9, 0.4, 0.07, 0.1])
        buttons["state_button"] = RadioButtons(state_ax, ('State 1', 'State 2', 'State 3'), active=state)
        buttons["state_button"].on_clicked(update_state)

    # Run Button
    if "run_button" not in buttons:
        run_ax = fig.add_axes([0.9, 0.3, 0.1, 0.05])
        buttons["run_button"] = Button(run_ax, f"Run iteration {iter}")
        buttons["run_button"].on_clicked(update_granule_stimulation_and_plots)

    # Error Button
    if "error_button" not in buttons:
        error_ax = fig.add_axes([0.9, 0.2, 0.1, 0.05])
        buttons["error_button"] = Button(error_ax, "Error")
        buttons["error_button"].on_clicked(update_inferior_olive_stimulation_and_plots)

    # Network Button
    if "network_button" not in buttons:
        network_ax = fig.add_axes([0.9, 0.1, 0.1, 0.05])
        buttons["network_button"] = Button(network_ax, "Show network")
        buttons["network_button"].on_clicked(toggle_network_graph)

    # Reset Button
    if "reset_button" not in buttons:
        reset_ax = fig.add_axes([0.9, 0.0, 0.1, 0.05])
        buttons["reset_button"] = Button(reset_ax, "Reset")
        buttons["reset_button"].on_clicked(reset)

    plt.draw()
    plt.pause(1)



def main():
    global t, granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, v_granule, v_purkinje, v_inferiorOlive, v_basket, t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np, fig, axes, iter
    init_variables()
    create_connections()
    #stimulate_granule_cell()
    [t, granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, v_granule, v_purkinje, v_inferiorOlive, v_basket] = recording()
    h.finitialize(-65)
    [t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np] = run_simulation(granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes)
    iter += 1
    
    


main()



try:
    while True:
        # Update the plot
        update_spike_and_weight_plot(t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np)
        time.sleep(2) # Delay between iterations

except KeyboardInterrupt:
    print("Simulation stopped by user.")
    plt.show()






