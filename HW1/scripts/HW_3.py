import numpy as np
import matplotlib.pyplot as plt

# 1. Constants
c_m = 1.0       # membrane capacitance (uF/cm^2)
g_L = 0.1       # leak conductance (mS/cm^2)
E_L = -65.0     # leak reversal potential (mV)
Ie = 3      # injected current (uA/cm^2)
dt = 0.01       # time step (ms)
T = 100         # total time (ms)

V_th = -50.0    # firing threshold
V_reset = -65.0 # reset voltage
Ref_period = 2  # refractory period (ms)

# 2. Time and voltage
time = np.arange(0, T, dt)
V = np.zeros(len(time))
V[0] = -70.0

spikes = []  # to store spike times

# 3. Simulation
    
Ref_period = False  
refractory_time = 2  
ref_count = 0  

for i in range(1, len(time)):
    if Ref_period:
        V[i] = V_reset
        ref_count += 1
        if ref_count >= refractory_time / dt: 
            Ref_period = False
            ref_count = 0 
    else:

        dVdt = (Ie - g_L * (V[i-1] - E_L)) / c_m
        V[i] = V[i-1] + dVdt * dt


        if V[i] >= V_th:
            spikes.append(time[i])
            V[i] = V_reset
            Ref_period = True  
            ref_count = 1  
        else:
            Ref_period = False


# 4. Plot
plt.figure(figsize=(10,4))
plt.plot(time, V, label="Membrane Potential")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.legend()
plt.show()

# Key differences between HH model and Noisy I&F:
#
# 1. HH model (deterministic): Current directly drives voltage above threshold.
#    - Sharp jump from 0 to ~55 Hz at a specific current threshold
#    - Shows discontinuous firing rate curve
#
# 2. Noisy I&F (stochastic): Random fluctuations occasionally exceed threshold.
#    - Gradual increase in firing with increasing noise
#    - Shows continuous firing rate curve starting from near-zero
#
# The HH model needs sufficient current to fire, while the noisy model can fire
# even with subthreshold average input due to random fluctuations.

