
import numpy as np  
import matplotlib.pyplot as plt 

E = -70 
Rm = 10 
tau = 10 
V_th = - 54 
V_reset = - 80

def inject_current(t):
    if 100 <= t < 400:  
        return 1.7  
    else:
        return 0.0

T = 500.0 
dt = 0.1  
t_steps = int(T/dt) 
time = np.arange(0, 500, dt)
spikes = np.zeros(t_steps) 

V = np.zeros(len(time))
V[0] = E 

# Run simulation
for i in range(1, len(time)):
    t = time[i]

    Ie = inject_current(t)
    
    dV_dt = (E - V[i-1] + Rm * Ie) / tau
    
    V[i] = V[i-1] + dV_dt * dt
    
    if V[i] >= V_th:
        spikes[i] = 1
        
        V[i] = V_reset


plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(time, V)
plt.axhline(y=V_th, color='r', linestyle='--', alpha=0.5, label='Threshold')
plt.ylabel('Membrane Potential (mV)')
plt.title('Integrate-and-Fire Neuron Model')
plt.legend()

plt.subplot(2, 1, 2)
current = np.array([inject_current(t) for t in time])
plt.plot(time, current)
plt.xlabel('Time (ms)')
plt.ylabel('Input Current (nA)')

plt.tight_layout()
plt.show()

