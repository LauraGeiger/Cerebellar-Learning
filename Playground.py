
import matplotlib.pyplot as plt
import numpy as np
from neuron import h, gui


'''

h.load_file("stdrun.hoc")

# Create cells
granule = h.Section(name="granule")
granule.insert("hh")
granule.L = granule.diam = 10

inferior_olive = h.Section(name="inferior_olive")
inferior_olive.insert("hh")
inferior_olive.L = inferior_olive.diam = 30

purkinje = h.Section(name="purkinje")
purkinje.insert("hh")
purkinje.L = purkinje.diam = 50

### 1. Simple Spikes: Granule Cell Stimulation ###
ss_stim = h.NetStim()
ss_stim.number = 1000   # High-frequency simple spikes
ss_stim.start = 10      # Start time (ms)
ss_stim.interval = 10   # 100 Hz firing

ss_syn = h.Exp2Syn(purkinje(0.5))
ss_syn.tau1 = 1
ss_syn.tau2 = 5
ss_syn.e = 0

g_nc = h.NetCon(ss_stim, ss_syn)
g_nc.weight[0] = 0.005
g_nc.delay = 3

### 2. Complex Spike: Inferior Olive Stimulation ###
iclamp = h.IClamp(inferior_olive(0.5))
iclamp.dur = 1e9  # Controlled by Vector.play()

# Define complex spike burst + pause
burst_start = 20
num_spikes = 5
burst_freq = 300
pause_duration = 100
pause_current = -0.3

burst_times = np.arange(burst_start, burst_start + (num_spikes * (1000 / burst_freq)), 1000 / burst_freq)
time_points = list(burst_times) + [burst_times[-1] + 1, burst_times[-1] + 1 + pause_duration]
current_values = [1.0] * len(burst_times) + [pause_current, 0]

t_vec = h.Vector(time_points)
i_vec = h.Vector(current_values)
i_vec.play(iclamp._ref_amp, t_vec, True)

### 3. Inferior Olive → Purkinje Connection (Overriding Effect) ###
io_syn = h.Exp2Syn(purkinje(0.5))
io_syn.tau1 = 1
io_syn.tau2 = 5
io_syn.e = 0

io_nc = h.NetCon(inferior_olive(0.5)._ref_v, io_syn, sec=inferior_olive)
io_nc.threshold = -20
io_nc.weight[0] = 0.1  # Stronger effect
io_nc.delay = 3

### 4. Record Voltage Traces ###
t = h.Vector()
v_granule = h.Vector()
v_olive = h.Vector()
v_purkinje = h.Vector()

t.record(h._ref_t)
v_granule.record(granule(0.5)._ref_v)
v_olive.record(inferior_olive(0.5)._ref_v)
v_purkinje.record(purkinje(0.5)._ref_v)

# Run simulation
h.finitialize(-65)
h.continuerun(200)


# Plot results
plt.figure(figsize=(8, 5))
plt.plot(t, v_granule, label="Granule Cell (Simple Spikes)", linestyle="dashed", color="blue")
plt.plot(t, v_olive, label="Inferior Olive (Complex Spike + Pause)", linestyle="dotted", color="green")
plt.plot(t, v_purkinje, label="Purkinje Cell", color="red")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Purkinje Cell Response to Granule & Inferior Olive Inputs")
plt.legend()
plt.show()
#'''

'''
from neuron import h
import numpy as np
import matplotlib.pyplot as plt

h.load_file("stdrun.hoc")

# Create cells
granule = h.Section(name="granule")
granule.insert("hh")
granule.L = granule.diam = 10

inferior_olive = h.Section(name="inferior_olive")
inferior_olive.insert("hh")
inferior_olive.L = inferior_olive.diam = 30

purkinje = h.Section(name="purkinje")
purkinje.insert("hh")
purkinje.L = purkinje.diam = 50

### 1. Simple Spikes: Granule Cell Stimulation via Current Injection ###
# Define current injection parameters for the granule cell
iclamp_granule = h.IClamp(granule(0.5))
iclamp_granule.delay = 10  # Start time (ms)
iclamp_granule.dur = 1     # Duration of current pulse (ms)
iclamp_granule.amp = 0.5   # Amplitude of current (nA)

# Set the current to pulse multiple times for simple spikes
# We will inject a current at regular intervals (simulating action potentials)
pulse_interval = 10  # Interval between pulses (ms)
pulse_num = 10       # Number of pulses

# Use a vector to inject multiple pulses
pulse_times = np.arange(10, 10 + pulse_num * pulse_interval, pulse_interval)  # Start at 10 ms
pulse_values = [0.5] * len(pulse_times)  # Current amplitude

# Set up the current injection using Vector
t_vec_granule = h.Vector(pulse_times)
i_vec_granule = h.Vector(pulse_values)
i_vec_granule.play(iclamp_granule._ref_amp, t_vec_granule, True)

### 2. Complex Spike: Inferior Olive Stimulation ###
iclamp_io = h.IClamp(inferior_olive(0.5))
iclamp_io.dur = 1e9  # Controlled by Vector.play()

# Define complex spike burst + pause
burst_start = 30  # Start later to see simple spikes first
num_spikes = 5
burst_freq = 300
pause_duration = 150  # Longer pause to suppress simple spikes
pause_current = -0.5  # Stronger hyperpolarization

burst_times = np.arange(burst_start, burst_start + (num_spikes * (1000 / burst_freq)), 1000 / burst_freq)
time_points = list(burst_times) + [burst_times[-1] + 1, burst_times[-1] + 1 + pause_duration]
current_values = [1.0] * len(burst_times) + [pause_current, 0]

t_vec_io = h.Vector(time_points)
i_vec_io = h.Vector(current_values)
i_vec_io.play(iclamp_io._ref_amp, t_vec_io, True)

### 3. Inferior Olive → Purkinje Connection (Overriding Effect) ###
io_syn = h.Exp2Syn(purkinje(0.5))
io_syn.tau1 = 1
io_syn.tau2 = 5
io_syn.e = 0

io_nc = h.NetCon(inferior_olive(0.5)._ref_v, io_syn, sec=inferior_olive)
io_nc.threshold = -20
io_nc.weight[0] = 0.1  # Stronger weight (dominates granule input)
io_nc.delay = 3

### 4. Granule → Purkinje Connection (Weaker Effect) ###
g_syn = h.Exp2Syn(purkinje(0.5))
g_syn.tau1 = 1
g_syn.tau2 = 5
g_syn.e = 0

g_nc = h.NetCon(granule(0.5)._ref_v, g_syn, sec=granule)
g_nc.threshold = -20
g_nc.weight[0] = 0.005  # Weaker weight
g_nc.delay = 3

### 5. Record Voltage Traces ###
t = h.Vector()
v_granule = h.Vector()
v_olive = h.Vector()
v_purkinje = h.Vector()

t.record(h._ref_t)
v_granule.record(granule(0.5)._ref_v)
v_olive.record(inferior_olive(0.5)._ref_v)
v_purkinje.record(purkinje(0.5)._ref_v)

# Run simulation
h.finitialize(-65)
h.continuerun(200)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(t, v_granule, label="Granule Cell (Simple Spikes)", linestyle="dashed", color="blue")
plt.plot(t, v_olive, label="Inferior Olive (Complex Spike + Pause)", linestyle="dotted", color="green")
plt.plot(t, v_purkinje, label="Purkinje Cell", color="red")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Purkinje Cell Response to Granule & Inferior Olive Inputs")
plt.legend()
plt.show()
'''
from neuron import h, gui
import numpy as np
import matplotlib.pyplot as plt

# === Create Cells ===
granule = h.Section(name="granule")  # Granule cell
granule.L = granule.diam = 10
granule.insert("hh")  # Active conductances

purkinje_cells = [h.Section(name=f"pc_{i}") for i in range(5)]  # 5 Purkinje cells
for pc in purkinje_cells:
    pc.L = pc.diam = 30
    pc.insert("hh")  # Hodgkin-Huxley for spiking

basket = h.Section(name="basket")  # Basket cell
basket.L = basket.diam = 10
basket.insert("hh")  # Active conductances

# === Excitatory Synapses (Granule → Purkinje) ===
exc_syns = [h.ExpSyn(pc(0.5)) for pc in purkinje_cells]
for syn in exc_syns:
    syn.tau = 5  # Synaptic decay time
    syn.e = 0    # Excitatory reversal potential

# === Inhibitory Synapses (Basket → 4 Purkinje) ===
inh_syns = [h.ExpSyn(purkinje_cells[i](0.5)) for i in range(4)]  # First 4 PCs inhibited
for syn in inh_syns:
    syn.tau = 5
    syn.e = -70  # Inhibitory reversal potential (GABA)

# === Direct Current Injection for Spiking ===
# Granule Cell (50 Hz spiking)
granule_stim = h.IClamp(granule(0.5))
granule_stim.delay = 10  # Start stimulation at 10 ms
granule_stim.dur = 1   # Duration of 200 ms
granule_stim.amp = 0.5   # Adjust to trigger 50 Hz spiking

# Basket Cell (Inhibition)
basket_stim = h.IClamp(basket(0.5))
basket_stim.delay = 10   # Start inhibition later
basket_stim.dur = 1    # Continuous inhibition
basket_stim.amp = 0.5    # Adjust for regular spiking

# === Connect Synapses ===
granule_netcons = [h.NetCon(granule(0.5)._ref_v, syn, sec=granule) for syn in exc_syns]
for nc in granule_netcons:
    nc.threshold = 10  # Spike detection threshold
    nc.weight[0] = 0.002  # Excitatory weight

basket_netcons = [h.NetCon(basket(0.5)._ref_v, syn, sec=basket) for syn in inh_syns]
for nc in basket_netcons:
    nc.threshold = 10  # Detect basket cell spikes
    nc.weight[0] = 0.05  # Strong inhibition

# === Recording Variables ===
t = h.Vector()  # Time
t.record(h._ref_t)

granule_spike_times = []  # Raster plot data
basket_spike_times = []

# Record Purkinje voltages
pc_voltages = [h.Vector() for _ in purkinje_cells]
for i, v in enumerate(pc_voltages):
    v.record(purkinje_cells[i](0.5)._ref_v)

# Record inhibitory synaptic currents
inh_currents = [h.Vector() for syn in inh_syns]
for i, v in enumerate(inh_currents):
    v.record(inh_syns[i]._ref_i)

# Function to record spike times
def record_granule_spikes():
    granule_spike_times.append(h.t)

def record_basket_spikes():
    basket_spike_times.append(h.t)

# Spike detectors
granule_spike_detector = h.NetCon(granule(0.5)._ref_v, None, sec=granule)
granule_spike_detector.threshold = -40
granule_spike_detector.record(record_granule_spikes)

basket_spike_detector = h.NetCon(basket(0.5)._ref_v, None, sec=basket)
basket_spike_detector.threshold = -40
basket_spike_detector.record(record_basket_spikes)

# === Run Simulation ===
h.finitialize(-65)
h.continuerun(200)

# Convert recorded spike times to NumPy arrays
granule_spike_times = np.array(granule_spike_times)
basket_spike_times = np.array(basket_spike_times)

# === Plot Results ===
fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

# 1. Purkinje Membrane Potentials
for i in range(5):
    axs[0].plot(t, pc_voltages[i], label=f"PC {i+1}")
axs[0].set_ylabel("Voltage (mV)")
axs[0].legend()
axs[0].set_title("Purkinje Cell Membrane Potentials")

# 2. Inhibitory Synaptic Currents
for i in range(4):
    axs[1].plot(t, inh_currents[i], label=f"PC {i+1} Inhibitory Input")
axs[1].set_ylabel("Current (nA)")
axs[1].legend()
axs[1].set_title("Inhibitory Synaptic Inputs (Basket → PC)")

# 3. Raster Plot of Granule Spikes
axs[2].scatter(granule_spike_times, np.ones_like(granule_spike_times), color="b", marker="|", s=100, label="Granule Spikes")
axs[2].set_ylabel("Granule")
axs[2].legend()
axs[2].set_title("Granule Cell Raster Plot")

# 4. Raster Plot of Basket Spikes
axs[3].scatter(basket_spike_times, np.ones_like(basket_spike_times), color="r", marker="|", s=100, label="Basket Spikes")
axs[3].set_xlabel("Time (ms)")
axs[3].set_ylabel("Basket")
axs[3].legend()
axs[3].set_title("Basket Cell Raster Plot")

plt.tight_layout()
plt.show()
