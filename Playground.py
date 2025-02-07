
import matplotlib.pyplot as plt
import numpy as np
from neuron import h, gui

'''
# Create multiple Purkinje Cells
num_cells = 3
cells = [h.Section(name=f'PC_{i}') for i in range(num_cells)]

# Insert necessary channels in each Purkinje cell
for cell in cells:
    cell.L = cell.diam = 20  # Soma size
    cell.insert('hh')  # Hodgkin-Huxley ion channels

# Create Parallel Fiber input shared by all Purkinje cells
pf_stim = h.NetStim()
pf_stim.number = 1000  # Continuous firing
pf_stim.start = 10    # Start time (ms)
pf_stim.interval = 20  # 50 Hz firing (Parallel Fiber input)

pf_syns = [h.ExpSyn(cell(0.5)) for cell in cells]
# Set common synapse properties
for pf_syn in pf_syns:
    pf_syn.tau = 2  # Fast decay (AMPA-like)
    pf_syn.e = 0    # Excitatory

# NetCon for Parallel Fiber input (connected to all Purkinje cells)
pf_ncs = [h.NetCon(pf_stim, pf_syn) for pf_syn in pf_syns]
#pf_ncs = h.NetCon(pf_stim, pf_syns)

def activate_GC_PC_synnapse(cell_ID):
    for ID, nc in enumerate(pf_ncs):
        if ID == cell_ID:
            print(f"Activate cell {ID}")
            nc.weight[0] = 0.005  # Small EPSP weight for active cell
            #nc.active(1)
        else:
            print(f"Deactivate cell {ID}")
            #nc.weight[0] = 0 #+ 0.001*ID # deactivate all others
            nc.active(0)

activate_GC_PC_synnapse(0)    # Activate Cell 1 and deactivate the others

# Record membrane potential of all cells
t_vec = h.Vector()
v_vecs = [h.Vector() for _ in range(num_cells)]

t_vec.record(h._ref_t)
for i in range(num_cells):
    v_vecs[i].record(cells[i](0.5)._ref_v)

# Run Simulation
h.tstop = 300  # Total simulation time (ms)
h.run()

# Plot Results
plt.figure(figsize=(8,6))
for i in range(num_cells):
    plt.plot(t_vec, v_vecs[i], label=f"Purkinje Cell {i}")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title("Selective Activation and Deactivation of Purkinje Cells")
plt.legend()
plt.show()
'''

'''
from neuron import h
import numpy as np
import matplotlib.pyplot as plt

h.load_file("stdrun.hoc")

# Create a single-compartment neuron
cell = h.Section(name="purkinje")
cell.insert("hh")  # Insert Hodgkin-Huxley channels for spiking
cell.L = cell.diam = 30  # Size

# Create an IClamp to inject current
iclamp = h.IClamp(cell(0.5))
iclamp.dur = 1e9  # Long duration; controlled by Vector.play()

# Define complex spike burst parameters
burst_start = 10  # ms
num_spikes = 5    # Number of spikes in burst
burst_freq = 300  # Hz (High frequency burst)

# Define pause parameters
pause_duration = 100  # ms
pause_current = -0.3  # nA (Hyperpolarizing pause)

# Generate burst times
burst_times = np.arange(burst_start, burst_start + (num_spikes * (1000 / burst_freq)), 1000 / burst_freq)

# Generate current waveform (burst followed by pause)
time_points = list(burst_times) + [burst_times[-1] + 1, burst_times[-1] + 1 + pause_duration]
current_values = [1.0] * len(burst_times) + [pause_current, 0]  # Spike burst (1 nA), then pause (-0.3 nA)

# Convert to NEURON Vectors
t_vec = h.Vector(time_points)
i_vec = h.Vector(current_values)

# Apply waveform to IClamp
i_vec.play(iclamp._ref_amp, t_vec, True)

# Run simulation
t = h.Vector()
v = h.Vector()
t.record(h._ref_t)
v.record(cell(0.5)._ref_v)

h.finitialize(-65)
h.continuerun(200)

# Plot results
plt.figure(figsize=(8,4))
plt.plot(t, v, label="Purkinje Cell Voltage")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Complex Spike and Pause")
plt.legend()
plt.show()
'''

'''
from neuron import h
import numpy as np
import matplotlib.pyplot as plt

h.load_file("stdrun.hoc")

# Create granule and Purkinje cells
granule = h.Section(name="granule")
granule.insert("hh")  # Hodgkin-Huxley spiking model
granule.L = granule.diam = 10  # Small cell

purkinje = h.Section(name="purkinje")
purkinje.insert("hh")  # Purkinje cell also has HH model
purkinje.L = purkinje.diam = 30  # Larger cell

# Create synapse from granule → Purkinje
syn = h.Exp2Syn(purkinje(0.5))
syn.tau1 = 1  # Synaptic rise time
syn.tau2 = 5  # Synaptic decay time
syn.e = 0  # Excitatory

# Create NetCon to link granule APs to the synapse
nc = h.NetCon(granule(0.5)._ref_v, syn, sec=granule)
nc.threshold = -20  # Spike detection threshold
nc.weight[0] = 0.01  # Synaptic weight
nc.delay = 3  # Transmission delay

# Create an IClamp for complex spike injection into the granule cell
iclamp = h.IClamp(granule(0.5))
iclamp.dur = 1e9  # Long duration; controlled by Vector.play()

# Define complex spike burst parameters
burst_start = 10  # Start time (ms)
num_spikes = 5    # Number of spikes in burst
burst_freq = 300  # Hz (High frequency burst)

# Define pause parameters
pause_duration = 100  # ms
pause_current = -0.3  # nA (Hyperpolarizing pause)

# Generate burst times
burst_times = np.arange(burst_start, burst_start + (num_spikes * (1000 / burst_freq)), 1000 / burst_freq)

# Generate current waveform (burst followed by pause)
time_points = list(burst_times) + [burst_times[-1] + 1, burst_times[-1] + 1 + pause_duration]
current_values = [1.0] * len(burst_times) + [pause_current, 0]  # Burst (1 nA), then pause (-0.3 nA)

# Convert to NEURON Vectors
t_vec = h.Vector(time_points)
i_vec = h.Vector(current_values)

# Apply waveform to IClamp
i_vec.play(iclamp._ref_amp, t_vec, True)

# Record simulation data
t = h.Vector()
v_granule = h.Vector()
v_purkinje = h.Vector()

t.record(h._ref_t)
v_granule.record(granule(0.5)._ref_v)
v_purkinje.record(purkinje(0.5)._ref_v)

# Run simulation
h.finitialize(-65)
h.continuerun(200)

# Plot results
plt.figure(figsize=(8, 4))
plt.plot(t, v_granule, label="Granule Cell Voltage", linestyle="dashed")
plt.plot(t, v_purkinje, label="Purkinje Cell Voltage", color="red")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Purkinje Cell Response to Granule Cell Complex Spike")
plt.legend()
plt.show()
'''

#'''
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