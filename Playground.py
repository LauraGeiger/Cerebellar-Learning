
import matplotlib.pyplot as plt
import numpy as np
from neuron import h, gui


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
