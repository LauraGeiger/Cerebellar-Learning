import numpy as np
import matplotlib.pyplot as plt
from neuron import h, gui


# Create a Purkinje Cell Soma
soma = h.Section(name='soma')
soma.L = soma.diam = 20  # Soma size
soma.insert('hh')  # Hodgkin-Huxley channels

# Parallel Fiber (PF) Stimulation - Simple Spikes
pf_stim = h.NetStim()
#pf_stim.number = 100  # Number of spikes
#pf_stim.start = 10    # Start time (ms)
#pf_stim.interval = 10  # Frequency 100 (Hz)

pf_syn = h.Exp2Syn(soma(0.5))
pf_syn.tau1 = 1  # Synaptic rise time
pf_syn.tau2 = 5  # Synaptic decay time
pf_syn.e = 0    # Excitatory

pf_nc = h.NetCon(pf_stim, pf_syn)
pf_nc.weight[0] = 0.01  # Small weight for simple spikes

# Climbing Fiber (CF) Stimulation - Complex Spikes
cf_stim = h.NetStim()
cf_stim.number = 1  # Single spike
cf_stim.start = 50  # Delayed onset
cf_burst_stim = h.NetStim()
cf_burst_stim.number = 3  # 3 burst spikes
cf_burst_stim.start = 51  # 1 ms after main CS
cf_burst_stim.interval = 10  # Fast intra-burst interval (3 ms)
cf_syn = h.Exp2Syn(soma(0.5))
cf_syn.tau1 = 10  # Synaptic rise time
cf_syn.tau2 = 50  # Synaptic decay time
cf_syn.e = 0

cf_nc = h.NetCon(cf_stim, cf_syn)
cf_nc.weight[0] = 0.1  # Large weight for complex spikes
cf_burst_nc = h.NetCon(cf_burst_stim, cf_syn)
cf_burst_nc.weight[0] = 0.05  # Weaker burst spikes

# Calcium-Activated K+ Current for Pause
ahp = h.IClamp(soma(0.5))  # Artificial hyperpolarizing current
ahp.delay = 52  # Shortly after complex spike
ahp.dur = 200   # Lasts for 200 ms
ahp.amp = 2  # Hyperpolarizing current (in nA)

# Recording Membrane Potential
t_vec = h.Vector()
v_vec = h.Vector()
t_vec.record(h._ref_t)
v_vec.record(soma(0.5)._ref_v)

# Run Simulation
h.finitialize(-65)
h.continuerun(400)
#h.tstop = 300  # Simulation time in ms
#h.run()

# Plot Results
plt.figure(figsize=(8,4))
plt.plot(t_vec, v_vec, label="Purkinje Cell Vm")
plt.axvline(50, color='red', linestyle='--', label="Complex Spike (CS)")
plt.axvspan(50, 250, color='red', alpha=0.2, label="Post-Complex Spike Pause")
plt.axvline(250, color='green', linestyle='--', label="Simple Spike Resumption")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title("Simple and Complex Spikes in a Purkinje Cell")
plt.legend()
plt.show()
