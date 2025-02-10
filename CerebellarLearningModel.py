from neuron import h, gui
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import time

from matplotlib.widgets import Button
from matplotlib.widgets import RadioButtons
from matplotlib.widgets import TextBox

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

# --- Create Network ---
num_granule = 3
num_purkinje = 5
num_inferior_olive = 1

granule_cells = [GranuleCell(i) for i in range(num_granule)]
purkinje_cells = [PurkinjeCell(i) for i in range(num_purkinje)]
inferior_olive_cells = [InferiorOliveCell(i) for i in range(num_inferior_olive)]

# --- STDP Parameters ---
tau_plus = 20  
tau_minus = 20  
A_plus = 0.005  
A_minus = 0.005
dt_LTP = 10  # Time window for LTP (ms)
dt_LTD = -10  # Time window for LTD (ms)
pf_initial_weight = 0.01
cf_initial_weight = 0.5
max_weight = 0.1
min_weight = 0.001



# --- Create Synapses and Connections ---
pf_syns = [[None for _ in range(num_purkinje)] for _ in range(num_granule)] # parallel fiber synapses
pf_ncs = [[None for _ in range(num_purkinje)] for _ in range(num_granule)] # parallel fiber netcons
cf_syns = [[None for _ in range(num_purkinje)] for _ in range(num_inferior_olive)] # climbing fiber synapses
cf_ncs = [[None for _ in range(num_purkinje)] for _ in range(num_inferior_olive)] # climbing fiber netcons

stimuli = []

def init_variables():
    global iter, state, mode, mode_dict, environment, frequency, weights, weights_over_time, processed_pairs, network_fig, network_ani, buttons

    iter = 0
    state = 1  # User can change this state (0, 1, or 2) based on desired behavior
    mode = 0
    mode_dict = {0:"Manual", 1:"Auto"}
    environment = {0:0, 1:2, 2:4} # "state:PC_ID" environment maps object_ID/state to the desired Purkinje Cell
    frequency = 50 # Hz
    weights = {}
    weights_over_time = { (pre_gid, post_gid): [] for pre_gid in range(num_granule) for post_gid in range(num_purkinje) } # track weights over time
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
            nc.weight[0] = pf_initial_weight
            nc.delay = 1
            pf_ncs[granule.gid][purkinje.gid] = nc
            weights[(granule.gid, purkinje.gid)] = pf_initial_weight + np.random.uniform(0,0.001)
            

    # Inferior Olive → Purkinje Connections (excitatory)
    for inferior_olive in inferior_olive_cells:
        for purkinje in purkinje_cells:
            syn = h.Exp2Syn(purkinje.soma(0.5))
            syn.e = 0  # Excitatory
            syn.tau1 = 5 # Synaptic rise time
            syn.tau2 = 25 # Synaptic decay time
            cf_syns[0][purkinje.gid] = syn
            nc = h.NetCon(inferior_olive.soma(0.5)._ref_v, syn, sec=inferior_olive.soma)
            nc.weight[0] = cf_initial_weight
            nc.delay = 1
            cf_ncs[inferior_olive.gid][purkinje.gid] = nc

def stimulate_highest_weight_PC(granule_gid):
    global last_activated_purkinje

    max_weight = -np.inf
    best_purkinje = None

    # Find the Purkinje cell with the highest weight
    for purkinje in purkinje_cells:
        try:
            print(f"PC{purkinje.gid+1}, voltage: {v_purkinje_np[purkinje.gid][-1]} mV")
            if v_purkinje_np[purkinje.gid][-1] > -55: # if membrane voltage is above 50 mV
                print(f"Skip PC{purkinje.gid+1}, voltage: {v_purkinje_np[purkinje.gid][-1]} mV")
                continue # Skip the blocked Purkinje cell
        except NameError: 
            None
        weight = weights[(granule_gid, purkinje.gid)]
        if weight > max_weight:
            max_weight = weight
            best_purkinje = purkinje
    

    try:
        print(f"Best purkinje: PC{best_purkinje.gid+1}")
        None
    except NameError: 
        print("v_purkinje_np not defined")
    except AttributeError:
        print("All purkinje cells blocked")

    
    g_id = state
    i_id = 0
    if best_purkinje != None:
        for p_id, purkinje in enumerate(purkinje_cells):
            if purkinje == best_purkinje:
                # Activate connections to purkinje cell with highest weight
                pf_ncs[g_id][p_id].weight[0] = pf_initial_weight
                cf_ncs[i_id][p_id].weight[0] = cf_initial_weight
            else:
                # Deactivate connections to all other purkinje cells
                pf_ncs[g_id][p_id].weight[0] = 0
                cf_ncs[i_id][p_id].weight[0] = 0
            
            last_activated_purkinje = best_purkinje  # Update last activated PC
            #print(f"Granule {granule_gid+1} spiked at {spike_time} → Triggering Purkinje {best_purkinje.gid+1} (weight {max_weight})")

def stimulate_granule_cell():
    # --- Stimulate Granule Cells Based on State ---
    g_id = state
    stim = h.IClamp(granule_cells[g_id].soma(0.5))
    stim.delay = 1/frequency*1000 * (iter + 1/2)
    stim.dur = 1
    stim.amp = 0.5
    stimuli.append(stim)
    print("Granule Spike simulated")

def update_granule_stimulation_and_plots(event=None):
    global t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, granule_spikes, purkinje_spikes, inferiorOlive_spikes, fig1, axes1, buttons, iter
    
    stimulate_granule_cell()
    [t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np] = run_simulation(granule_spikes, purkinje_spikes, inferiorOlive_spikes)
    iter += 1
    buttons["run_button"].label.set_text(f"Run iteration {iter}")
    [fig1, axes1] = update_spike_and_weight_plot(t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, fig1, axes1)

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


def update_inferior_olive_stimulation_and_plots(event=None):
    global t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, granule_spikes, purkinje_spikes, inferiorOlive_spikes, fig1, axes1, buttons
    
    stimulate_inferior_olive_cell()
    [t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np] = run_simulation(granule_spikes, purkinje_spikes, inferiorOlive_spikes, error=True)
    [fig1, axes1] = update_spike_and_weight_plot(t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, fig1, axes1)
    
    if buttons["network_button"].label.get_text() == "Hide network":
        update_and_draw_network() # Update network if open


# Update state variable
def update_state(event):
    global state, buttons
    for i in range(3):
        if buttons["state_button"].value_selected == f"State {i+1}":
            state = i




def toggle_mode(event=None):
    global state, mode, mode_dict, simulation_completed

    mode = next(i for i, value in mode_dict.items() if value == buttons["automatic_button"].value_selected)

    # Toggle button visibilities
    #buttons["state_button"].ax.set_visible(True if mode == 0 else False)
    buttons["run_button"].ax.set_visible(True if mode == 0 else False)
    buttons["error_button"].ax.set_visible(True if mode == 0 else False)
    buttons["network_button"].ax.set_visible(True if mode == 0 else False)
    buttons["reset_button"].ax.set_visible(True if mode == 0 else False)


    # Trigger error automatically
    if mode == 1: # automatic mode
        initial_state = state
        
        #if last_activated_purkinje.gid != environment[state]:
        #    print(f"{h.t} PC{last_activated_purkinje.gid+1} not desired, triggering error")
        #    update_inferior_olive_stimulation_and_plots() # automatically trigger error
                    
        #while True:
        for i in range(10):
            update_granule_stimulation_and_plots()
            if simulation_completed == True:
                simulation_completed = False
                if last_activated_purkinje.gid != environment[state]:
                    print(f"{h.t} PC{last_activated_purkinje.gid+1} not desired, triggering error")
                    update_inferior_olive_stimulation_and_plots() # automatically trigger error
                    time.sleep(5) # Delay between iterations
                time.sleep(5) # Delay between iterations
            if mode == 0:
                break

            #state = (state + 1) % 3
            #buttons["state_button"].set_active(state)
            #if state == initial_state:
            #    break
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

def show_network_graph():
    global network_fig, network_ani, spike_times, G, edges, node_colors_list, node_pos, network_ax

    network_fig = plt.figure()
    network_ax = network_fig.add_subplot(111)
    G = nx.DiGraph()

    # --- Define Nodes ---
    granule_nodes = [f"GC{i+1}" for i in range(num_granule)]
    purkinje_nodes = [f"PC{i+1}" for i in range(num_purkinje)]
    G.add_nodes_from(granule_nodes, color="blue")
    G.add_nodes_from(purkinje_nodes, color="red")
    G.add_node("IO", color="green")

    # --- Define Node Positions ---
    node_pos = {g: (0, i+1) for i, g in enumerate(granule_nodes)}  # Granule Cells at x = 0
    node_pos.update({p: (1, i) for i, p in enumerate(purkinje_nodes)})  # Purkinje Cells at x = 1
    node_pos["IO"] = (2, len(purkinje_nodes) // 2)  # Inferio Olive Cell at x = 2

    # --- Define Node Colors
    node_colors = {node: "blue" if node.startswith("G") else "red" for node in G.nodes}
    node_colors["IO"] = "green"
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
    
def recording():
    # --- Record Spiking Activity and Voltages---
    t = h.Vector().record(h._ref_t)
    granule_spikes = {i: h.Vector() for i in range(num_granule)}
    purkinje_spikes = {i: h.Vector() for i in range(num_purkinje)}
    inferiorOlive_spikes = {i: h.Vector() for i in range(num_inferior_olive)}
    v_granule = {i: h.Vector().record(granule_cells[i].soma(0.5)._ref_v) for i in range(num_granule)}
    v_purkinje = {i: h.Vector().record(purkinje_cells[i].soma(0.5)._ref_v) for i in range(num_purkinje)}
    v_inferiorOlive = {i: h.Vector().record(inferior_olive_cells[i].soma(0.5)._ref_v) for i in range(num_inferior_olive)}

    for i, granule in enumerate(granule_cells):
        nc = h.NetCon(granule.soma(0.5)._ref_v, None, sec=granule.soma)
        nc.threshold = -20
        nc.record(granule_spikes[i])

    for i, purkinje in enumerate(purkinje_cells):
        nc = h.NetCon(purkinje.soma(0.5)._ref_v, None, sec=purkinje.soma)
        nc.threshold = -20
        nc.record(purkinje_spikes[i])

    for i, inferior_olive in enumerate(inferior_olive_cells):
        nc = h.NetCon(inferior_olive.soma(0.5)._ref_v, None, sec=inferior_olive.soma)
        nc.threshold = -20
        nc.record(inferiorOlive_spikes[i])

    return [t, granule_spikes, purkinje_spikes, inferiorOlive_spikes, v_granule, v_purkinje, v_inferiorOlive]

def run_simulation(granule_spikes, purkinje_spikes, inferiorOlive_spikes, error = False):
    global iter, spike_times, processed_pairs, frequency, simulation_completed

    print("Run simulation")
    if error:
        time_span = 1/2 * 1/frequency*1000
        stop_time = h.t + time_span
    else:
        stop_time = 1/frequency*1000 * (iter + 1) # run 20 ms per iteration

    # Continuously run the simulation and update weights during the simulation
    while h.t < stop_time: 
        h.continuerun(h.t + 1)  # Incrementally run the simulation
        
        # --- Trigger Purkinje Cell Spike ---
        for g_id in range(num_granule):
            for pre_t in granule_spikes[g_id]:
                if pre_t > h.t -1:
                    stimulate_highest_weight_PC(g_id)

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


    # --- Convert Spike Data ---
    spike_times = {f"GC{i+1}": list(granule_spikes[i]) for i in range(num_granule)}
    spike_times.update({f"PC{i+1}": list(purkinje_spikes[i]) for i in range(num_purkinje)})
    spike_times.update({f"IO{i+1}": list(inferiorOlive_spikes[i]) for i in range(num_inferior_olive)})

    # --- Convert Voltage Data and Weights ---
    t_np = np.array(t)
    v_granule_np = np.array([vec.to_python() for vec in v_granule.values()])
    v_purkinje_np = np.array([vec.to_python() for vec in v_purkinje.values()])
    v_inferiorOlive_np = np.array([vec.to_python() for vec in v_inferiorOlive.values()])

    simulation_completed = True
    return [t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np]

def update_spike_and_weight_plot(t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, fig1 = None, axes1 = None):
    global buttons
    print("Update Spike and weight Plot")

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
    for granule in granule_cells:
        
        ax1 = axes1[0, granule.gid]
        ax1.set_title(f"GC{granule.gid+1} Spiking Activity")
        ax1.plot(t_np, v_granule_np[granule.gid], label=f"GC{granule.gid+1}", color="blue")
        ax1.plot(t_np, v_inferiorOlive_np[io_id], label=f"IO", color="black")

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

    #fig1.canvas.draw_idle()
    plt.draw()
    plt.pause(1)

    return [fig1, axes1]


def main():
    global t, granule_spikes, purkinje_spikes, inferiorOlive_spikes, v_granule, v_purkinje, v_inferiorOlive, t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, fig1, axes1, iter
    init_variables()
    create_connections()
    stimulate_granule_cell()
    [t, granule_spikes, purkinje_spikes, inferiorOlive_spikes, v_granule, v_purkinje, v_inferiorOlive] = recording()
    h.finitialize(-65)
    [t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np] = run_simulation(granule_spikes, purkinje_spikes, inferiorOlive_spikes)
    iter += 1
    

    ## Trigger error automatically
    #if mode == 1: # automatic mode
    #    if last_activated_purkinje.gid != environment[state]:
    #        print(f"{h.t} PC{last_activated_purkinje.gid+1} not desired, triggering error")
    #        update_inferior_olive_stimulation_and_plots() # automatically trigger error
    
    
    

main()

try:
    while True:
        # Update the plot
        fig1, axes1 = update_spike_and_weight_plot(t_np, v_granule_np, v_purkinje_np, v_inferiorOlive_np, fig1, axes1)
        time.sleep(2) # Delay between iterations

except KeyboardInterrupt:
    print("Simulation stopped by user.")
    plt.show()






