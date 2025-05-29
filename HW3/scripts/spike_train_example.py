import numpy as np
import matplotlib.pyplot as plt

# Parameters
r = 10               # Firing rate in Hz
T = 1000             # Total simulation time in ms
dt = 0.1             # Time step in ms
dt_sec = dt / 1000   # Convert dt from ms to seconds

# Calculate time points
time = np.arange(0, T, dt)
n_steps = len(time)

# Initialize arrays to store spikes and probabilities
pre_spikes = np.zeros(n_steps)       # Presynaptic spike train (0 or 1)
P_depressing = np.ones(n_steps)      # Release probability for depressing synapse
P_facilitating = np.zeros(n_steps)   # Release probability for facilitating synapse
trans_depressing = np.zeros(n_steps) # Transmissions from depressing synapse
trans_facilitating = np.zeros(n_steps) # Transmissions from facilitating synapse

# Parameters for synapses
tau_P_depressing = 300     # Time constant for depressing synapse in ms
P0_depressing = 1.0        # Resting release probability for depressing synapse

tau_P_facilitating = 100   # Time constant for facilitating synapse in ms
P0_facilitating = 0.0      # Resting release probability for facilitating synapse

# Generate presynaptic spike train (Poisson process)
for i in range(1, n_steps):
    # Generate a spike with probability r*dt (convert dt to seconds)
    spike_probability = r * dt_sec
    if np.random.rand() < spike_probability:
        pre_spikes[i] = 1
        
        # When presynaptic spike occurs, check for transmission at each synapse
        # For depressing synapse
        if np.random.rand() < P_depressing[i-1]:
            trans_depressing[i] = 1
            P_depressing[i] = 0  # Reset P to 0 after release
        else:
            P_depressing[i] = P_depressing[i-1]
            
        # For facilitating synapse
        if np.random.rand() < P_facilitating[i-1]:
            trans_facilitating[i] = 1
        # P always increases for facilitating synapse when presynaptic spike occurs
        P_facilitating[i] = P_facilitating[i-1] + 0.1*(1 - P_facilitating[i-1])
    else:
        # No presynaptic spike, update P values according to differential equation
        # For depressing synapse: τ_P * dP/dt = P0 - P
        P_depressing[i] = P_depressing[i-1] + dt/tau_P_depressing * (P0_depressing - P_depressing[i-1])
        
        # For facilitating synapse: τ_P * dP/dt = P0 - P
        P_facilitating[i] = P_facilitating[i-1] + dt/tau_P_facilitating * (P0_facilitating - P_facilitating[i-1])
        
        # No transmissions without presynaptic spike
        trans_depressing[i] = 0
        trans_facilitating[i] = 0

# Create plots
plt.figure(figsize=(10, 8))

# Plot 1: Presynaptic spikes
plt.subplot(5, 1, 1)
plt.plot(time, pre_spikes, 'k|', markersize=10)
plt.ylabel('Presynaptic\nspikes')
plt.title('Spike Train and Synaptic Transmission')

# Plot 2: Depressing synapse transmissions
plt.subplot(5, 1, 2)
plt.plot(time, trans_depressing, 'r|', markersize=10)
plt.ylabel('Depressing\ntransmissions')

# Plot 3: Facilitating synapse transmissions
plt.subplot(5, 1, 3)
plt.plot(time, trans_facilitating, 'g|', markersize=10)
plt.ylabel('Facilitating\ntransmissions')

# Plot 4: P for depressing synapse
plt.subplot(5, 1, 4)
plt.plot(time, P_depressing)
plt.ylabel('P (depressing)')

# Plot 5: P for facilitating synapse
plt.subplot(5, 1, 5)
plt.plot(time, P_facilitating)
plt.xlabel('Time (ms)')
plt.ylabel('P (facilitating)')

plt.tight_layout()
plt.show()
