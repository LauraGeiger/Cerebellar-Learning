import numpy as np
import matplotlib.pyplot as plt
from neuron import h, gui

# Create a single-compartment soma
soma = h.Section(name="soma")
soma.L = soma.diam = 30  # Set diameter and length
soma.insert("hh")  # Hodgkin-Huxley model

# Synapses for Simple Spikes (SS - Parallel Fibers)
ss_syn = h.Exp2Syn(soma(0.5))
ss_syn.tau1, ss_syn.tau2, ss_syn.e = 0.5, 2.0, 0.0  # AMPA-like

# Synapses for Complex Spikes (CS - Climbing Fibers)
cs_syn = h.Exp2Syn(soma(0.5))
cs_syn.tau1, cs_syn.tau2, cs_syn.e = 1.0, 10.0, 0.0  # NMDA-like

# --- Simple Spikes Before CS ---
ss_stim1 = h.NetStim()
ss_stim1.number = 50  # Fires from 10 ms to 50 ms
ss_stim1.start = 10
ss_stim1.interval = 10  # 100 Hz firing

# --- Complex Spike ---
cs_stim = h.NetStim()
cs_stim.number = 1  # Single burst onset
cs_stim.start = 50  # First CS at 50 ms

# --- Burst generator for Complex Spike ---
burst_gen = h.NetStim()
burst_gen.number = 3  # 3 burst spikes
burst_gen.start = 51  # 1 ms after main CS
burst_gen.interval = 3  # Fast intra-burst interval (3 ms)

# --- Simple Spikes After CS ---
ss_stim2 = h.NetStim()
ss_stim2.number = 50  # Fires from 100 ms onward (ensuring enough time for repolarization)
ss_stim2.start = 100
ss_stim2.interval = 10  # 100 Hz firing

# --- Connect Simple Spikes ---
nc_ss1 = h.NetCon(ss_stim1, ss_syn)
nc_ss1.weight[0] = 0.002  # Small EPSP

nc_ss2 = h.NetCon(ss_stim2, ss_syn)
nc_ss2.weight[0] = 0.002  # Continue after CS

# --- Connect Complex Spike ---
nc_cs = h.NetCon(cs_stim, cs_syn)
nc_cs.weight[0] = 0.05  # Large input

nc_burst = h.NetCon(burst_gen, cs_syn)
nc_burst.weight[0] = 0.02  # Weaker burst spikes

# --- Recording Variables ---
t_vec = h.Vector()  # Time
v_vec = h.Vector()  # Voltage
t_vec.record(h._ref_t)
v_vec.record(soma(0.5)._ref_v)

# --- Monitor Membrane Potential (to ensure full repolarization) ---
v_monitor = h.Vector()
v_monitor.record(soma(0.5)._ref_v)

# Run Simulation
h.finitialize(-65)
h.continuerun(200)

# --- Plot Results ---
plt.figure(figsize=(8, 4))
plt.plot(t_vec, v_vec, label="Membrane Potential (mV)", color='black')
plt.axvline(50, color='red', linestyle='--', label="Complex Spike (CS)")
plt.axvline(100, color='blue', linestyle='--', label="Simple Spike Resumption")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title("Purkinje Cell Simple & Complex Spikes")
plt.legend()
plt.show()
