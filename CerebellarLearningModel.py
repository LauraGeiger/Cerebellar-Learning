from neuron import h, gui
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from matplotlib.widgets import RadioButtons
from matplotlib.patches import FancyArrow, FancyArrowPatch
#from scipy.interpolate import make_curve
import networkx as nx
import time






# using Python 3.8.20

# --- Variable Initialization for plotting
plt.ion()  # Turn on interactive mode
fig1, axes1 = None, None

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
    global iter, state, mode, mode_dict, environment, frequency, weights, weights_over_time, processed_GC_spikes, processed_pairs, network_fig, network_ani, buttons

    iter = 0
    state = 1  # User can change this state (0, 1, or 2) based on desired behavior
    mode = 0
    mode_dict = {0:"Manual", 1:"Auto"}
    environment = {0:0, 1:2, 2:4} # "state:PC_ID" environment maps object_ID/state to the desired Purkinje Cell
    frequency = 50 # Hz
    weights = {}
    weights_over_time = { (pre_gid, post_gid): [] for pre_gid in range(num_granule) for post_gid in range(num_purkinje) } # track weights over time
    processed_GC_spikes = { (g_gid): set() for g_gid in range(num_granule)} # store the processed granule cell spikes
    processed_pairs = { (pre_id, post_id): set() for pre_id in range(num_granule) for post_id in range(num_purkinje) } # store the processed spike pairs for each (pre_id, post_id)
    network_fig = None
    network_ani = None
    buttons = {}


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
            #weights[(granule.gid, purkinje.gid)] = pf_initial_weight + np.random.uniform(0,0.001)
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
        #weight = weights[(granule_gid, purkinje.gid)]
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
    global t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np, granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, fig1, axes1, buttons, iter
    g_id = state
    activate_highest_weight_PC(g_id)

    stimulate_granule_cell()
    [t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np] = run_simulation(granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes)
    iter += 1
    buttons["run_button"].label.set_text(f"Run iteration {iter}")
    [fig1, axes1] = update_spike_and_weight_plot(t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np, fig1, axes1)

    if buttons["network_button"].label.get_text() == "Hide network":
        update_and_draw_network() # Update network if open


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
    global t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np, granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, fig1, axes1, buttons
    
    stimulate_inferior_olive_cell()
    [t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np] = run_simulation(granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, error=True)
    [fig1, axes1] = update_spike_and_weight_plot(t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np, fig1, axes1)
    
    if buttons["network_button"].label.get_text() == "Hide network":
        update_and_draw_network() # Update network if open


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
    global network_fig, buttons
    if buttons["network_button"].label.get_text() == "Hide network":
        buttons["network_button"].label.set_text("Show network")
        plt.close(network_fig)
    else:
        buttons["network_button"].label.set_text("Hide network")
        show_network_graph()

def update_and_draw_network():
    global G, edges, node_colors_list, node_pos, network_ax
    # --- Define Edge Weights---
    edge_weights = []
    for i in range(num_granule):
        for j in range(num_purkinje):
            weight = weights[(i, j)]  # Get the latest weight value
            edge_weights.append(weight)
    for j in range(num_purkinje):
        edge_weights.append(0.01)  # Default weight for IO connections
    G.add_edges_from(edges)

    # --- Normalize Edge Widths ---
    min_w, max_w = min(edge_weights), max(edge_weights)
    if max_w > min_w:  # Avoid division by zero
        edge_widths = [(w - min_w) / (max_w - min_w) * 5 + 1 for w in edge_weights]  # Scale to range 1-6
    else:
        edge_widths = [2 for _ in edge_weights]  # Default width if all weights are the same

    # --- Define Edge Labels ---
    edge_labels = {(f"GC{i+1}", f"PC{j+1}"): f"{weights[(granule_cells[i].gid, purkinje_cells[j].gid)]:.3f}"
        for i in range(num_granule) for j in range(num_purkinje)}

    # --- Drawing ---
    nx.draw(G, node_pos, with_labels=True, node_color=node_colors_list, edge_color="gray", ax=network_ax,
                node_size=1000, font_size=12, font_weight="bold", arrows=True, width=edge_widths)
    nx.draw_networkx_edge_labels(G, node_pos, edge_labels=edge_labels, ax=network_ax, font_size=10, font_weight="bold", label_pos=0.2)



def draw_purkinje(ax, x, y, width=0.2, color='orange'):
    """Draws a Purkinje neuron with dendrites and a separate soma."""
    # Soma (neuron body)
    ax.scatter(x, y[0], s=200, color=color)
    
    # Dendritic tree
    ax.plot([x, x], [y[0], y[-1]], color=color, lw=4)  # Main trunk
    for i in range(1,num_granule+1):  # Branching
        ax.plot([x, x - width], [y[i], y[i] + width], color=color, lw=2)
        ax.plot([x, x + width], [y[i], y[i] + width], color=color, lw=2)

def draw_parallel_fiber(ax, x, y, length=4, height=1.5):
    """Draws a parallel fiber extending across Purkinje cells."""
    ax.plot([x - length / 4, x + length], [y , y], color='green', lw=2, linestyle='dashed')

def draw_granule_to_parallel(ax, x, y_start, y_end):
    """Draws a granule cell axon that ascends vertically and forms a parallel fiber."""
    ax.plot([x, x], [y_start, y_end], color='blue', lw=2)  # Vertical axon
    draw_parallel_fiber(ax, x, y_end)  # Horizontal fiber

def draw_climbing_fiber(ax, x, y_start, y_end):
    """Draws a climbing fiber from the Inferior Olive wrapping around a Purkinje cell."""
    t = np.linspace(0, 1, 100)
    x_vals = x + 0.2 * np.sin(6 * np.pi * t)  # Wavy pattern for wrapping
    y_vals = y_start + (y_end - y_start) * t
    ax.plot(x_vals, y_vals, color='red', lw=2, label="Climbing Fiber")

def draw_basket_cell(ax, x_start, x_end, y):
    """Draws a basket cell connecting to all Purkinje cells at their soma."""
    ax.scatter(x_start, y, s=150, color='purple', edgecolors='black', label="Basket Cell")
    ax.plot([x_start , x_end], [y, y], color='purple', lw=2)

def show_network_graph():
    fig, ax = plt.subplots(figsize=(8, 6))


    purkinje_x = np.linspace(0, 2, num_purkinje)
    granule_x = np.linspace(-2, -1, num_granule)
    olive_x = np.linspace(0, 2, num_inferior_olive)
    basket_x = -0.5
    
    purkinje_y = np.linspace(0, 1.5, num_granule+1)
    granule_y = -2  # Bottom row
    basket_y = purkinje_y[0]
    olive_y = -3  # Inferior Olive position

    # Draw Inferior Olive cell
    ax.scatter(olive_x, olive_y, s=200, color='red', edgecolors='black', label="Inferior Olive")

    # Draw Purkinje cells
    for x in purkinje_x:
        draw_purkinje(ax, x, purkinje_y, width=0.2)
        draw_climbing_fiber(ax, x, olive_y, purkinje_y[-1])  # Climbing fibers

    # Draw Granule cells, vertical axons, and parallel fibers
    for i, x in enumerate(granule_x):
        ax.scatter(x, granule_y, color='blue', s=100, label="Granule Cell") 
        draw_granule_to_parallel(ax, x, granule_y, purkinje_y[i+1])

    # Draw Basket cell connecting to Purkinje cell somas
    draw_basket_cell(ax, basket_x, purkinje_x[-1], basket_y)

    # Labels
    ax.text(purkinje_x[-1] + 0.2, purkinje_y[0], "Purkinje Cells", fontsize=12, color="orange")
    ax.text(granule_x[0], granule_y - 0.2, "Granule Cells", fontsize=12, color="blue")
    ax.text(granule_x[0], purkinje_y[-1] + 0.2, "Parallel Fibers", fontsize=12, color="green")
    ax.text(olive_x, olive_y - 0.2, "Inferior Olive", fontsize=12, color="red")
    ax.text(basket_x, basket_y + 0.2, "Basket Cell", fontsize=12, color="purple")

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3.5, 3)
    ax.axis('off')
    plt.title("Realistic Cerebellar Network")
    #plt.legend(loc="upper right", fontsize=10)
    plt.show()

show_network_graph()



def old_show_network_graph():
    global network_fig, network_ani, spike_times, G, edges, node_colors_list, node_pos, network_ax

    network_fig = plt.figure()
    network_ax = network_fig.add_subplot(111)
    G = nx.DiGraph()

    # --- Define Nodes ---
    granule_nodes = [f"GC{i+1}" for i in range(num_granule)]
    purkinje_nodes = [f"PC{i+1}" for i in range(num_purkinje)]
    G.add_nodes_from(granule_nodes, color="blue")
    G.add_nodes_from(purkinje_nodes, color="green")
    G.add_node("IO", color="red")

    # --- Define Node Positions ---
    node_pos = {g: (0, i+1) for i, g in enumerate(granule_nodes)}  # Granule Cells at x = 0
    node_pos.update({p: (1, i) for i, p in enumerate(purkinje_nodes)})  # Purkinje Cells at x = 1
    node_pos["IO"] = (2, len(purkinje_nodes) // 2)  # Inferio Olive Cell at x = 2

    # --- Define Node Colors
    node_colors = {node: "blue" if node.startswith("G") else "green" for node in G.nodes}
    node_colors["IO"] = "red"
    node_colors_list = [node_colors[node] for node in G.nodes]

    # --- Define Edges ---
    edges = []
    for i in range(num_granule):
        for j in range(num_purkinje):
            edges.append((f"GC{i+1}", f"PC{j+1}"))
    for j in range(num_purkinje):
        edges.append(("IO", f"PC{j+1}"))
    G.add_edges_from(edges)

    update_and_draw_network()
    
    '''
    # --- Animation ---
    def update(frame):
        ax.clear()
        current_time = frame
        active_nodes = [node for node in G.nodes if any(abs(sp - current_time) < 2 for sp in spike_times[node])]
        colors = ["yellow" if node in active_nodes else node_colors[node] for node in G.nodes]

        # Add edge labels for weights
        edge_labels = {(f"GC{i+1}", f"PC{j+1}"): f"{weights[(granule_cells[i].gid, purkinje_cells[j].gid)]:.3f}"
                for i in range(num_granule) for j in range(num_purkinje)}

        nx.draw(G, pos, with_labels=True, node_color=colors, edge_color="gray", ax=ax,
                node_size=1000, font_size=12, font_weight="bold", arrows=True, width=edge_widths)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=10, font_weight="bold", label_pos=0.2)

    network_ani = animation.FuncAnimation(network_fig, update, frames=np.arange(0, 100, 1), interval=100)
    '''  


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

def update_spike_and_weight_plot(t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np, fig1 = None, axes1 = None):
    global buttons
    
    if fig1 is None or axes1 is None:
        fig1, axes1 = plt.subplots(2, num_granule, figsize=(5 * num_granule, 8), sharex=True)
        # Share y-axis within each row
        for row in range(2):
            for col in range(1, num_granule):  # Start from second column
                axes1[row, col].sharey(axes1[row, 0])  # Share y-axis with first column
    else:
        # Clear previous plots
        axes1 = np.array(axes1)
        for row in range(2):
            for col in range(num_granule):
                axes1[row,col].cla()

    io_id = 0
    b_id = 0
    for granule in granule_cells:
        
        ax1 = axes1[0, granule.gid]
        ax1.set_title(f"GC{granule.gid+1} Spiking Activity")
        ax1.plot(t_np, v_granule_np[granule.gid], label=f"GC{granule.gid+1}", color="blue")
        ax1.plot(t_np, v_inferiorOlive_np[io_id], label=f"IO", color="black")
        ax1.plot(t_np, v_basket_np[b_id], label=f"B", color="pink")

        ax2 = axes1[1, granule.gid]
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
            
        ax1.legend()
        ax2.legend()

    # Label y-axes only on the first column
    axes1[0, 0].set_ylabel("Membrane Voltage (mV)")
    axes1[1, 0].set_ylabel("Synaptic Weight")


    # --- Button ---

    # Automatic Button
    if "automatic_button" not in buttons:
        automatic_ax = fig1.add_axes([0.9, 0.7, 0.07, 0.1])
        buttons["automatic_button"] = RadioButtons(automatic_ax, (mode_dict[0], mode_dict[1]), active=mode)
        buttons["automatic_button"].on_clicked(toggle_mode)

    # State Button
    if "state_button" not in buttons:
        state_ax = fig1.add_axes([0.9, 0.6, 0.07, 0.1])
        buttons["state_button"] = RadioButtons(state_ax, ('State 1', 'State 2', 'State 3'), active=state)
        buttons["state_button"].on_clicked(update_state)

    # Run Button
    if "run_button" not in buttons:
        run_ax = fig1.add_axes([0.9, 0.5, 0.1, 0.05])
        buttons["run_button"] = Button(run_ax, f"Run iteration {iter}")
        buttons["run_button"].on_clicked(update_granule_stimulation_and_plots)

    # Error Button
    if "error_button" not in buttons:
        error_ax = fig1.add_axes([0.9, 0.4, 0.1, 0.05])
        buttons["error_button"] = Button(error_ax, "Error")
        buttons["error_button"].on_clicked(update_inferior_olive_stimulation_and_plots)

    # Network Button
    if "network_button" not in buttons:
        network_ax = fig1.add_axes([0.9, 0.3, 0.1, 0.05])
        buttons["network_button"] = Button(network_ax, "Show network")
        buttons["network_button"].on_clicked(toggle_network_graph)

    # Reset Button
    if "reset_button" not in buttons:
        reset_ax = fig1.add_axes([0.9, 0.2, 0.1, 0.05])
        buttons["reset_button"] = Button(reset_ax, "Reset")
        buttons["reset_button"].on_clicked(reset)

    plt.draw()
    plt.pause(1)

    return [fig1, axes1]


def main():
    global t, granule_spikes, purkinje_spikes, inferiorOlive_spikes, basket_spikes, v_granule, v_purkinje, v_inferiorOlive, v_basket, t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np, fig1, axes1, iter
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
        fig1, axes1 = update_spike_and_weight_plot(t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, v_basket_np, fig1, axes1)
        time.sleep(2) # Delay between iterations

except KeyboardInterrupt:
    print("Simulation stopped by user.")
    plt.show()






