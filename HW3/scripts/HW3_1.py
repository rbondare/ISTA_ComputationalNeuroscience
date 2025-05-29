import numpy as np
import matplotlib.pyplot as plt


c_m = 10  # nF/mm2
g_L = 1.0  # uS/mm2
E_L = -70  # mV
E_ex = 0  # mV
V_th = -54  # mV
V_reset = -80  # mV
tau_ex = 5  # ms 

dt = 0.1  # ms
T = 1000  # ms
time = np.arange(0, T, dt)
n_steps = len(time)


V = np.zeros(n_steps)
g_ex = np.zeros(n_steps)
I_ex = np.zeros(n_steps) 
spikes = [] 


V[0] = E_L
g_ex[0] = 0

A_LTP = 0.35
A_LTD = 0.4
t_LTP = 25 
t_LTD = 35 
delta_g_us = 1.2 
delta_g_cs = 0 


us = np.array([100, 200, 200, 300, 400, 500, 600]) 
cs = np.array([90, 190, 290, 390, 490, 590, 690, 790, 890])
last_cs_time = 0 

for i in range(1, n_steps):
    t = time[i]
    g_ex[i] = g_ex[i-1] * (1 - dt/tau_ex)  # Euler approximation of exponential decay of ex conductance
    
    # for unconditioned stimulus 
    if np.min(np.abs(t - us)) < dt/2:
        g_ex[i] += delta_g_us

    # for conditioned stimulus
    elif np.min(np.abs(t - cs)) < dt/2:
        last_cs_time = t
    
        # LTD implementation - if there have been any postsynaptic spikes
        if len(spikes) > 0:
            delta_t_LTD = t - spikes[-1]  # Time since last postsynaptic spike
            if delta_t_LTD > 0:  # If presynaptic spike follows postsynaptic spike
                delta_g_cs = delta_g_cs - A_LTD * np.exp(-delta_t_LTD / t_LTD)
    
        g_ex[i] += delta_g_cs
    
        if g_ex[i] > 1.2:
            g_ex[i] = 1.2
        elif g_ex[i] < 0:
            g_ex[i] = 0
            
    I_leak = g_L * (V[i-1] - E_L)
    I_ex[i] = g_ex[i] * (V[i-1] - E_ex)
    
    dV_dt = (-I_leak - I_ex[i]) / c_m
    V[i] = V[i-1] + dV_dt * dt
            
    if V[i] >= V_th:
        spikes.append(t)
        V[i] = V_reset
        
        # LTP implementation - if CS occurred before this spike
        if last_cs_time > 0 and t > last_cs_time:
            delta_t_LTP = t - last_cs_time
            delta_g_cs = delta_g_cs + A_LTP * np.exp(-delta_t_LTP / t_LTP)
            
            # Fixed indentation of these bounds checks
            if delta_g_cs > 1.2:
                delta_g_cs = 1.2
            elif delta_g_cs < 0:
                delta_g_cs = 0

plt.figure(figsize=(12, 8))


plt.subplot(211)
plt.plot(time, V)

for spike in spikes:
    plt.axvline(x=spike, color='k', linestyle='-', alpha=0.3)

plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.title('Integrate and Fire with STDP')
plt.legend()

plt.subplot(212)
plt.plot(time, g_ex)

 # Mark US events
for u in us:
    plt.axvline(x=u, color='r', linestyle='--', alpha=0.7, label='US' if u==us[0] else "")
    
# Mark CS events
for c in cs:
    plt.axvline(x=c, color='g', linestyle='--', alpha=0.7, label='CS' if c==cs[0] else "")

plt.xlabel('Time (ms)')
plt.ylabel('Conductance (uS/mm²)')
plt.legend()

plt.tight_layout()
#plt.savefig("Integrate and_Fire STDP.png")
plt.show()