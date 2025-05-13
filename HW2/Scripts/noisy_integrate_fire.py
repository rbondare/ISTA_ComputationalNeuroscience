import numpy as np
import matplotlib.pyplot as plt


tau_m = 10
V_th = -54 
V_reset = -80  
E_base = -56.0  


dt = 0.1  
T = 1000
time = np.arange(0, T, dt)
n_steps = len(time)

# Range of noise levels 
sigma_V_range = np.linspace(0, 10, 20)  # From 0 to 10 mV

def simulate_neuron(sigma_V, disable_spikes=False):
    """
        disable_spikes: If True, spike mechanism will be disabled (for part a)
    """
    V = np.zeros(n_steps)
    spikes = np.zeros(n_steps)
    V[0] = E_base
    actual_V_th = V_th if not disable_spikes else 1000
    noise_term = sigma_V * np.sqrt(2 * tau_m / dt) * np.random.randn(n_steps)
    E_eff = E_base + noise_term
    
    for i in range(1, n_steps):
        dV_dt = (-V[i-1] + E_eff[i]) / tau_m
        V[i] = V[i-1] + dV_dt * dt
        if V[i] >= actual_V_th:
            spikes[i] = 1
            V[i] = V_reset
    return V, spikes, E_eff

# For part (b): Compute firing rates for different noise levels
firing_rates = []
for sigma_V in sigma_V_range:

    _, spikes, _ = simulate_neuron(sigma_V, disable_spikes=False)
    firing_rate = np.sum(spikes) / (T / 1000)  # Convert to frequency 
    firing_rates.append(firing_rate)


plt.figure(figsize=(10, 6))
plt.plot(sigma_V_range, firing_rates, 'o-', color='blue')
plt.xlabel('Noise Level sigma_V (mV)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Firing Rate vs. Noise Level')
plt.savefig('HW2_Q2_b.png')
plt.show()


std_devs = []
for sigma_V in sigma_V_range:
    V, _, _ = simulate_neuron(sigma_V, disable_spikes=True)  
    std_dev = np.std(V)
    std_devs.append(std_dev)

plt.figure(figsize=(10, 6))
plt.plot(sigma_V_range, std_devs, 'b-', label='Standard Deviation of V')
plt.plot(sigma_V_range, sigma_V_range, 'g--', label='theoretical', alpha=0.5)
plt.xlabel('Input Noise Level sigma (mV)')
plt.ylabel('Standard Deviation of V (mV)')
plt.title('Standard Deviation of Membrane Potential vs. Noise Level')
plt.legend()
plt.savefig('HW2_Q2_a.png')
plt.show()

