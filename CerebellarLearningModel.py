from neuron import h, gui
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx

from matplotlib.widgets import Button

# using Python 3.8.20

class STDP: # Spike timing dependent plasticity
    def __init__(self, netcon):
        self.netcon = netcon
        self.weight = netcon.weight[0]  # Synaptic weight
        self.pre_times = []
        self.post_times = []
        self.dt_LTP = 10  # Time window for LTP (ms)
        self.dt_LTD = -10  # Time window for LTD (ms)
        self.A_plus = 0.005  # LTP increment
        self.A_minus = -0.005  # LTD decrement
        self.tau_stdp = 20 # range of pre-to-postsynaptic interspike intervals over which synaptic strengthening or weakening occurs
        self.weight_changes = []

    def pre_spike(self, t):
        #print(f"Pre-synaptic spike at {t} ms")
        self.pre_times.append(t)
        self.update_weights()

    def post_spike(self, t):
        #print(f"Post-synaptic spike at {t} ms")
        self.post_times.append(t)
        self.update_weights()

    def update_weights(self):
        if len(self.pre_times) > 0 and len(self.post_times) > 0:
            dt = self.post_times[-1] - self.pre_times[-1]  # Time difference

            if dt > 0 and dt < self.dt_LTP:  # LTP Condition
                dW = self.A_plus * (max_weight - self.weight) # dependent on current weight
                #dW = self.A_plus * np.exp(dt / self.tau_stdp) # dependent on timing
                print("LTP: ", dW)
                self.weight += dW
            elif dt < 0 and dt > self.dt_LTD:  # LTD Condition
                dW = self.A_minus * self.weight # dependet on current weight
                #dW = self.A_minus * np.exp(dt / self.tau_stdp) # dependent on timing
                print("LTD: ", dW)
                self.weight += dW

            self.weight = max(0, min(self.weight, max_weight))  # Keep weight within limits
            self.netcon.weight[0] = self.weight
            self.weight_changes.append((h.t, self.weight))
'''
# Create pre- and post-synaptic neurons
pre_neuron = h.Section(name='pre_neuron')
post_neuron = h.Section(name='post_neuron')

h.topology()
h.load_file("nrngui.hoc")
shape_window = h.PlotShape(True)  # Enable shape visualization
shape_window.show(0)  # Show all sections

# Set neuron properties
for sec in [pre_neuron, post_neuron]:
    sec.L = sec.diam = 20  # Soma
    sec.insert('hh')  # Hodgkin-Huxley channels

# Create a synapse on the post-neuron
syn = h.ExpSyn(post_neuron(0.5))
#syn.tau = 2  # Rise time (ms)
#syn.tau2 = 5  # Decay time (ms)
syn.e = 0  # Reversal potential (mV)

# Connect the neurons with a NetCon
initial_weight = 0.01
max_weight = 0.1
delay = 5 # Delay in ms
pre_netcon = h.NetCon(pre_neuron(0.5)._ref_v, syn, sec=pre_neuron)
pre_netcon.weight[0] = initial_weight  # Initial synaptic weight
pre_netcon.delay = delay
post_netcon = h.NetCon(post_neuron(0.5)._ref_v, syn, sec=post_neuron)
post_netcon.weight[0] = initial_weight  # Initial synaptic weight
post_netcon.delay = delay



stdp = STDP(pre_netcon)

# Record spikes
pre_spikes = h.Vector()
post_spikes = h.Vector()

pre_netcon.record(pre_spikes)
post_netcon.record(post_spikes)

# Stimulate Pre-Synaptic Neuron
stim_pre = h.IClamp(pre_neuron(0.5))
stim_pre.delay = 20
stim_pre.dur = 1
stim_pre.amp = 0.5

stim_pre2 = h.IClamp(pre_neuron(0.5))
stim_pre2.delay = 42
stim_pre2.dur = 1
stim_pre2.amp = 0.5

stim_pre3 = h.IClamp(pre_neuron(0.5))
stim_pre3.delay = 61
stim_pre3.dur = 1
stim_pre3.amp = 0.5

# Stimulate Post-Synaptic Neuron (to induce pairing)
stim_post = h.IClamp(post_neuron(0.5))
stim_post.delay = 15  # Post fires shortly after Pre to induce LTP
stim_post.dur = 1
stim_post.amp = 0.5

stim_post2 = h.IClamp(post_neuron(0.5))
stim_post2.delay = 45
stim_post2.dur = 1
stim_post2.amp = 0.5

stim_post3 = h.IClamp(post_neuron(0.5))
stim_post3.delay = 56
stim_post3.dur = 1
stim_post3.amp = 0.5

# Record voltage and weight changes
t = h.Vector().record(h._ref_t)
v_pre = h.Vector().record(pre_neuron(0.5)._ref_v)
v_post = h.Vector().record(post_neuron(0.5)._ref_v)


# Run simulation
h.finitialize(-65)

# Continuously run the simulation and update weights during the simulation
while h.t < 100:
    h.continuerun(h.t + 1)  # Incrementally run the simulation
    # Call pre_spike and post_spike whenever spikes are detected
    for t_spike in pre_spikes:
        #print("t_spike ", t_spike)
        #print("h.t ", h.t)
        if (t_spike > h.t-1): # only process newly detected spikes
          stdp.pre_spike(t_spike)

    for t_spike in post_spikes:
        if (t_spike > h.t-1): # only process newly detected spikes
          stdp.post_spike(t_spike)

# Convert data to NumPy arrays
t_np = np.array(t)
v_pre_np = np.array(v_pre)
v_post_np = np.array(v_post)
weights = np.array(stdp.weight_changes)
print(stdp.weight_changes)

print(f"size t_np: {len(t_np)}, size weights: {len(weights)}")

# Add initial weight at time 0ms until (first weight_change - delay)
init_weight_start = np.array([[0, initial_weight]])
init_weight_end = np.array([[weights[0,0] - delay, initial_weight]])
weights = np.vstack([init_weight_start, init_weight_end, weights])

print("Pre-spike times:", list(pre_spikes))
print("Post-spike times:", list(post_spikes))
print("Weights:", weights)

# Plot Voltage Traces
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
ax1.plot(t_np, v_pre_np, label="Pre-synaptic Neuron")
ax1.plot(t_np, v_post_np, label="Post-synaptic Neuron")
ax1.legend()
ax1.set_ylabel("Membrane Voltage (mV)")
ax1.set_title("Neuronal Spiking")

# Plot Synaptic Weight Changes
if len(weights) > 0:
  ax2.plot(weights[:,0], weights[:,1])
  ax2.set_xlabel("Time (ms)")
  #ax2.set_ylim(bottom=0.0098)
  #ax2.set_ylim(top=0.0101)
  ax2.set_ylabel("Synaptic Weight")
  ax2.set_title("STDP-Induced Synaptic Plasticity")
else:
    print("No weight changes recorded!")

#plt.tight_layout()
plt.show()
'''

'''
# --- Create Granule Cell Class ---
class GranuleCell:
    def __init__(self):
        self.soma = h.Section(name='granule_cell')
        self.soma.L = self.soma.diam = 10  # Small soma for granule cells
        self.soma.insert('hh')  # Hodgkin-Huxley conductances

# --- Create Purkinje Cell Class ---
class PurkinjeCell:
    def __init__(self):
        self.soma = h.Section(name='purkinje_cell')
        self.soma.L = self.soma.diam = 50  # Larger soma for Purkinje cells
        self.soma.insert('hh')  # Hodgkin-Huxley channels

# --- Create Network ---
granule_cells = [GranuleCell() for _ in range(3)]  # 3 Granule Cells
purkinje_cells = [PurkinjeCell() for _ in range(5)]  # 5 Purkinje Cells

# --- Connect Granule Cells to Purkinje Cells ---
synapses = []
netcons = []
for granule in granule_cells:
    for purkinje in purkinje_cells:
        syn = h.ExpSyn(purkinje.soma(0.5))  # Synapse on Purkinje cell
        syn.e = 0  # Excitatory synapse (reversal potential)
        syn.tau = 2  # Synaptic time constant

        nc = h.NetCon(granule.soma(0.5)._ref_v, syn, sec=granule.soma)
        nc.weight[0] = 0.01  # Initial synaptic weight
        nc.delay = 5  # Synaptic delay

        synapses.append(syn)
        netcons.append(nc)

h.topology()

# --- Stimulate Granule Cells ---
stimuli = []
for i, granule in enumerate(granule_cells):
    stim = h.IClamp(granule.soma(0.5))
    stim.delay = 10 + i * 5  # Staggered activation
    stim.dur = 1
    stim.amp = 0.5
    stimuli.append(stim)

# --- Record Activity ---
t = h.Vector().record(h._ref_t)
granule_voltages = [h.Vector().record(g.soma(0.5)._ref_v) for g in granule_cells]
purkinje_voltages = [h.Vector().record(p.soma(0.5)._ref_v) for p in purkinje_cells]

# --- Run Simulation ---
h.finitialize(-65)
h.continuerun(100)

# --- Plot Results ---
plt.figure(figsize=(10, 6))

# Plot Granule Cell Activity
for i, v in enumerate(granule_voltages):
    plt.plot(t, v, label=f"Granule Cell {i+1}")

# Plot Purkinje Cell Activity
for i, v in enumerate(purkinje_voltages):
    plt.plot(t, v, linestyle="dashed", label=f"Purkinje Cell {i+1}")

plt.xlabel("Time (ms)")
plt.ylabel("Membrane Voltage (mV)")
plt.title("Granule-Purkinje Network Activity")
plt.legend()
plt.show()
'''

'''
# --- Create Granule and Purkinje Cell Classes ---
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

# --- Create Network ---
num_granule = 3
num_purkinje = 5

granule_cells = [GranuleCell(i) for i in range(num_granule)]
purkinje_cells = [PurkinjeCell(i) for i in range(num_purkinje)]

# --- Create Synapses and Connections ---
synapses = []
netcons = []
for granule in granule_cells:
    for purkinje in purkinje_cells:
        syn = h.ExpSyn(purkinje.soma(0.5))
        syn.e = 0
        syn.tau = 2
        nc = h.NetCon(granule.soma(0.5)._ref_v, syn, sec=granule.soma)
        nc.weight[0] = 0.01
        nc.delay = 5
        synapses.append(syn)
        netcons.append(nc)

# --- Stimulate Granule Cells ---
stimuli = []
for i, granule in enumerate(granule_cells):
    stim = h.IClamp(granule.soma(0.5))
    stim.delay = 10 + i * 5
    stim.dur = 1
    stim.amp = 0.5
    stimuli.append(stim)

# --- Record Spiking Activity ---
t = h.Vector().record(h._ref_t)
granule_spikes = [h.Vector() for _ in granule_cells]
purkinje_spikes = [h.Vector() for _ in purkinje_cells]

for i, granule in enumerate(granule_cells):
    nc = h.NetCon(granule.soma(0.5)._ref_v, None, sec=granule.soma)
    nc.threshold = -20  # Detect spikes
    nc.record(granule_spikes[i])

for i, purkinje in enumerate(purkinje_cells):
    nc = h.NetCon(purkinje.soma(0.5)._ref_v, None, sec=purkinje.soma)
    nc.threshold = -20
    nc.record(purkinje_spikes[i])

# --- Run Simulation ---
h.finitialize(-65)
h.continuerun(100)

# --- Convert Spike Data ---
spike_times = {f"G{i+1}": list(granule_spikes[i]) for i in range(num_granule)}
spike_times.update({f"P{i+1}": list(purkinje_spikes[i]) for i in range(num_purkinje)})

# --- Create NetworkX Graph ---
G = nx.DiGraph()
granule_nodes = [f"GC{i+1}" for i in range(num_granule)]
purkinje_nodes = [f"PC{i+1}" for i in range(num_purkinje)]
G.add_nodes_from(granule_nodes, color="blue")
G.add_nodes_from(purkinje_nodes, color="red")
edges = [(g, p) for g in granule_nodes for p in purkinje_nodes]
G.add_edges_from(edges)

# --- Define Positions for Graph ---
pos = {g: (0, i) for i, g in enumerate(granule_nodes)}
pos.update({p: (2, i - 1) for i, p in enumerate(purkinje_nodes)})

# --- Animation Function ---
fig, ax = plt.subplots(figsize=(8, 6))
node_colors = {node: "blue" if node.startswith("G") else "red" for node in G.nodes}

def update(frame):
    ax.clear()
    current_time = frame  # Time in ms
    active_nodes = [node for node in G.nodes if any(abs(sp - current_time) < 2 for sp in spike_times[node])]
    
    colors = [("yellow" if node in active_nodes else node_colors[node]) for node in G.nodes]
    
    nx.draw(G, pos, with_labels=True, node_color=colors, edge_color="gray",
            node_size=1000, font_size=12, font_weight="bold", arrows=True)

    ax.set_title(f"Neural Network Activity at {current_time:.1f} ms")

# --- Run Animation ---
ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 100, 1), interval=100)

plt.show()
'''


state = 3  # User can change this state (1, 2, or 3) based on desired behavior
environment = {1:1, 2:3, 3:5} # "state:PC_ID" environment maps object_ID/state to the desired Purkinje Cell

# --- STDP Parameters ---
tau_plus = 20  
tau_minus = 20  
A_plus = 0.005  
A_minus = 0.005
dt_LTP = 10  # Time window for LTP (ms)
dt_LTD = -10  # Time window for LTD (ms)
initial_weight = 0.01
max_weight = 0.1
min_weight = 0.001

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
    def __init__(self):
        self.soma = h.Section(name='inferior_olive')
        self.soma.L = self.soma.diam = 20
        self.soma.insert('hh')

# --- Create Network ---
num_granule = 3
num_purkinje = 5

granule_cells = [GranuleCell(i) for i in range(num_granule)]
purkinje_cells = [PurkinjeCell(i) for i in range(num_purkinje)]
inferior_olive = InferiorOliveCell()

# --- Create Synapses and Connections ---
synapses = []
netcons = []
error = False
weights = {}
#for granule in granule_cells:
#    for purkinje in purkinje_cells:
#        weights[(granule.gid, purkinje.gid)] = {"times": [], "weights": []}


# Granule → Purkinje Connections (excitatory)
for granule in granule_cells:
    for purkinje in purkinje_cells:
        syn = h.ExpSyn(purkinje.soma(0.5))
        syn.e = 0
        #syn.tau = 2
        nc = h.NetCon(granule.soma(0.5)._ref_v, syn, sec=granule.soma)
        nc.weight[0] = initial_weight + np.random.uniform(0,0.001)
        nc.delay = 3
        synapses.append(syn)
        netcons.append(nc)
        weights[(granule.gid, purkinje.gid)] = nc.weight[0] 

# Inferior Olive → Purkinje Connections (inhibitory)
for purkinje in purkinje_cells:
    syn = h.ExpSyn(purkinje.soma(0.5))
    syn.e = -70  # Inhibitory reversal potential
    syn.tau = 3
    nc = h.NetCon(inferior_olive.soma(0.5)._ref_v, syn, sec=inferior_olive.soma)
    #nc.weight[0] = initial_weight
    nc.delay = 3
    synapses.append(syn)
    netcons.append(nc)
    #weights[("IO", purkinje.gid)] = initial_weight



# --- Stimulate Granule Cells Based on State ---
stimuli = []
if state == 1:   stim = h.IClamp(granule_cells[0].soma(0.5))
elif state == 2: stim = h.IClamp(granule_cells[1].soma(0.5))
elif state == 3: stim = h.IClamp(granule_cells[2].soma(0.5))
stim.delay = 10
stim.dur = 1
stim.amp = 0.5
stimuli.append(stim)

def trigger_spike_at_highest_weight_PC(granule_gid, spike_time):
    max_weight = -np.inf
    best_purkinje = None

    # Find the Purkinje cell with the highest weight
    for purkinje in purkinje_cells:
        weight = weights[(granule_gid, purkinje.gid)]
        if weight > max_weight:
            max_weight = weight
            best_purkinje = purkinje

    if best_purkinje:
        # Inject a small current to trigger a spike
        stim = h.IClamp(best_purkinje.soma(0.5))
        stim.delay = spike_time + 1  # Small delay after spike
        stim.dur = 1
        stim.amp = 2
        stimuli.append(stim)
        print(f"Granule {granule_gid+1} spiked at {spike_time} → Triggering Purkinje {best_purkinje.gid+1} (weight {max_weight})")

# Stimulate Inferior Olive if previous activated PC resulted in an error
def trigger_spike_IO(event):
    stim = h.IClamp(inferior_olive.soma(0.5))
    stim.delay = h.t + 1
    stim.dur = 1
    stim.amp = 2
    stimuli.append(stim)
    print(f"Inferior Olive spike triggered at time {h.t + 1} ms")

# --- STDP Update Function ---
def update_weights(pre_gid, post_gid, delta_t, t):
    if delta_t > 0 and delta_t < dt_LTP:  
        dw = A_plus * np.exp(-delta_t / tau_plus)
    elif delta_t < 0 and delta_t > dt_LTD:  
        dw = -A_minus * np.exp(delta_t / tau_minus)
    else: dw = 0
    new_weight = weights[(pre_gid, post_gid)] + dw
    weights[(pre_gid, post_gid)] = np.clip(new_weight, min_weight, max_weight)




# --- Record Spiking Activity and Voltages---
t = h.Vector().record(h._ref_t)
granule_spikes = {i: h.Vector() for i in range(num_granule)}
purkinje_spikes = {i: h.Vector() for i in range(num_purkinje)}
inferiorOlive_spikes = h.Vector()
v_granule = {i: h.Vector().record(granule_cells[i].soma(0.5)._ref_v) for i in range(num_granule)}
v_purkinje = {i: h.Vector().record(purkinje_cells[i].soma(0.5)._ref_v) for i in range(num_purkinje)}
V_inferiorOlive = h.Vector().record(inferior_olive.soma(0.5)._ref_v)

granule_spike_detectors = []
for i, granule in enumerate(granule_cells):
    nc = h.NetCon(granule.soma(0.5)._ref_v, None, sec=granule.soma)
    nc.threshold = -20
    nc.record(granule_spikes[i])

for i, purkinje in enumerate(purkinje_cells):
    nc = h.NetCon(purkinje.soma(0.5)._ref_v, None, sec=purkinje.soma)
    nc.threshold = -20
    nc.record(purkinje_spikes[i])

nc_io = h.NetCon(inferior_olive.soma(0.5)._ref_v, None, sec=inferior_olive.soma)
nc_io.threshold = -20
nc_io.record(inferiorOlive_spikes)


# Initialize a dictionary to track weights over time
weights_over_time = { (pre_gid, post_gid): [] for pre_gid in range(num_granule) for post_gid in range(num_purkinje) }


# --- Run Simulation ---
h.finitialize(-65)

# Initialize a dictionary to store the processed spike pairs for each (pre_id, post_id)
processed_pairs = { (pre_id, post_id): set() for pre_id in range(num_granule) for post_id in range(num_purkinje) }

# Continuously run the simulation and update weights during the simulation
while h.t < 50:
    h.continuerun(h.t + 1)  # Incrementally run the simulation
    
    # --- Trigger Purkinje Cell Spike ---
    for pre_id in range(num_granule):
        for pre_t in granule_spikes[pre_id]:
            if pre_t > h.t -1:
                trigger_spike_at_highest_weight_PC(pre_id, pre_t)

    # --- Apply STDP ---
    for pre_id in range(num_granule):
        for post_id in range(num_purkinje):
            for pre_t in granule_spikes[pre_id]:
                for post_t in purkinje_spikes[post_id]:
                    if (pre_t, post_t) not in processed_pairs[(pre_id, post_id)]:
                        delta_t = post_t - pre_t
                        update_weights(pre_id, post_id, delta_t, h.t)
                        print(f"update weights for GC{pre_id+1} <-> PC{post_id+1} pre_t {pre_t} post_t {post_t}")
                        processed_pairs[(pre_id, post_id)].add((pre_t, post_t))

            # Track the weight at the current time step
            while len(weights_over_time[(pre_id, post_id)]) < len(t):
                weights_over_time[(pre_id, post_id)].append(weights[(pre_id, post_id)])
            
# --- Convert Spike Data ---
spike_times = {f"GC{i+1}": list(granule_spikes[i]) for i in range(num_granule)}
spike_times.update({f"PC{i+1}": list(purkinje_spikes[i]) for i in range(num_purkinje)})
spike_times["IO"] = list(inferiorOlive_spikes)

# --- Convert Voltage Data and Weights ---
t_np = np.array(t)
v_granule_np = np.array([vec.to_python() for vec in v_granule.values()])
v_purkinje_np = np.array([vec.to_python() for vec in v_purkinje.values()])
v_inferiorOlive_np = np.array(V_inferiorOlive.to_python())

# --- Plot Voltage and Weight Traces ---
fig1, axes = plt.subplots(2, num_granule, figsize=(5*num_granule, 8), sharex=True)

# Share y-axis within each row
for row in range(2):
    for col in range(1, num_granule):  # Start from second column
        axes[row, col].sharey(axes[row, 0])  # Share y-axis with first column

for gc_id in range(num_granule):
    # --- Spiking Plot for GC and its connected PCs ---
    ax1 = axes[0, gc_id]
    ax1.plot(t_np, v_granule_np[gc_id], label=f"GC{gc_id+1}", color="blue")
    for pc_id in range(num_purkinje):
        ax1.plot(t_np, v_purkinje_np[pc_id], label=f"PC{pc_id+1}", linestyle="dashed")
    ax1.set_title(f"GC{gc_id+1} Spiking Activity")
    ax1.legend()

    # --- Weight Plot for GC to all connected PCs ---
    ax2 = axes[1, gc_id]
    for pc_id in range(num_purkinje):
        if len(weights_over_time[(gc_id, pc_id)]) > 0:
            ax2.plot(t_np, weights_over_time[(gc_id, pc_id)], label=f"PC{pc_id+1}")

    ax2.set_xlabel("Time (ms)")
    ax2.set_title(f"GC{gc_id+1} Synaptic Weights")
    ax2.legend()

# Label y-axes only on the first column
axes[0, 0].set_ylabel("Membrane Voltage (mV)")
axes[1, 0].set_ylabel("Synaptic Weight")


# --- Button ---
fig3 = plt.figure()
ax_btn = fig3.add_axes([0.7, 0.05, 0.1, 0.075])  # [left, bottom, width, height]
button = Button(ax_btn, "Stimulate IO")
button.on_clicked(trigger_spike_IO)


# --- Create NetworkX Graph ---
fig2, ax = plt.subplots(figsize=(8, 6))
G = nx.DiGraph()
granule_nodes = [f"GC{i+1}" for i in range(num_granule)]
purkinje_nodes = [f"PC{i+1}" for i in range(num_purkinje)]
G.add_nodes_from(granule_nodes, color="blue")
G.add_nodes_from(purkinje_nodes, color="red")
G.add_node("IO", color="green")

# --- Define Edges ---
edges = []
edge_weights = []
for i in range(num_granule):
    for j in range(num_purkinje):
        weight = weights[(i, j)]  # Get the latest weight value
        edges.append((f"GC{i+1}", f"PC{j+1}"))
        edge_weights.append(weight)
for j in range(num_purkinje):
    edges.append(("IO", f"PC{j+1}"))
    edge_weights.append(0.01)  # Default weight for IO connections
G.add_edges_from(edges)

# --- Normalize Edge Widths ---
min_w, max_w = min(edge_weights), max(edge_weights)
if max_w > min_w:  # Avoid division by zero
    edge_widths = [(w - min_w) / (max_w - min_w) * 5 + 1 for w in edge_weights]  # Scale to range 1-6
else:
    edge_widths = [2 for _ in edge_weights]  # Default width if all weights are the same

# --- Animation ---
node_colors = {node: "blue" if node.startswith("G") else "red" for node in G.nodes}
node_colors["IO"] = "green"

# --- Define Positions ---
pos = {g: (0, i+1) for i, g in enumerate(granule_nodes)}  # Granule Cells at x = 0
pos.update({p: (1, i) for i, p in enumerate(purkinje_nodes)})  # Purkinje Cells at x = 1
pos["IO"] = (2, len(purkinje_nodes) // 2)  # Inferio Olive Cell at x = 2

# --- Animation ---
node_colors = {node: "blue" if node.startswith("G") else "red" for node in G.nodes}
node_colors["IO"] = "green"

def update(frame):
    ax.clear()
    current_time = frame
    active_nodes = [node for node in G.nodes if any(abs(sp - current_time) < 2 for sp in spike_times[node])]
    colors = ["yellow" if node in active_nodes else node_colors[node] for node in G.nodes]

    # Add edge labels for weights
    edge_labels = {(f"GC{i+1}", f"PC{j+1}"): f"{weights[(granule_cells[i].gid, purkinje_cells[j].gid)]:.3f}"
               for i in range(num_granule) for j in range(num_purkinje)}

    nx.draw(G, pos, with_labels=True, node_color=colors, edge_color="gray",
            node_size=1000, font_size=12, font_weight="bold", arrows=True, width=edge_widths)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_weight="bold", label_pos=0.2)

ani = animation.FuncAnimation(fig2, update, frames=np.arange(0, 100, 1), interval=100)



#plt.tight_layout()
plt.show()

