import numpy as np
import matplotlib.pyplot as plt


c_m = 10  # nF/mm2
g_L = 1.0  # uS/mm2
E_L = -70  # mV
E_ex = 0  # mV
V_th = -54  # mV
V_reset = -80  # mV
tau_ex = 10  # ms
delta_g_ex = 0.5  # uS/mm2

dt = 0.1  # ms
T = 500  # ms
time = np.arange(0, T, dt)
n_steps = len(time)


V = np.zeros(n_steps)
g_ex = np.zeros(n_steps)
I_ex = np.zeros(n_steps) 


V[0] = E_L
g_ex[0] = 0


presynaptic_spikes = np.array([100, 200, 230, 300, 320, 400, 410]) 


for i in range(1, n_steps):
    t = time[i]
    g_ex[i] = g_ex[i-1] * (1 - dt/tau_ex)  # Euler approximation of exponential decay
    
    if np.min(np.abs(t - presynaptic_spikes)) < dt/2:
        g_ex[i] += delta_g_ex
    
    I_leak = g_L * (V[i-1] - E_L)
    I_ex[i] = g_ex[i] * (V[i-1] - E_ex)
    
    dV_dt = (-I_leak - I_ex[i]) / c_m
    V[i] = V[i-1] + dV_dt * dt

    if V[i] >= V_th:
        V[i] = V_reset


plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(time, V)
plt.axhline(y=V_th, color='r', linestyle='-', label='Threshold')
plt.ylabel('Membrane Potential (mV)')
plt.title('Integrate-and-Fire Neuron with Excitatory Synaptic Conductance')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, I_ex)
plt.xlabel('Time (ms)')
plt.ylabel('Synaptic Current (nA)')
plt.title('Synaptic Current vs Time')
for spike in presynaptic_spikes:
    plt.axvline(x=spike, color='g', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('HW2_Q1.png')
plt.show()

