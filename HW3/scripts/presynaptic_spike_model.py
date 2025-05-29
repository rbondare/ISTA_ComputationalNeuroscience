import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
T = 2000            # Total simulation time in ms
dt = 0.1            # Time step in ms
dt_sec = dt / 1000  # Convert dt from ms to seconds
time = np.arange(0, T, dt)
n_steps = len(time)

# Firing rate parameter
r = 20              # Firing rate in Hz - you can change this value

# Synapse parameters
tau_P_depressing = 300     # Time constant for depressing synapse in ms
P0_depressing = 1.0        # Resting release probability for depressing synapse

tau_P_facilitating = 100   # Time constant for facilitating synapse in ms
P0_facilitating = 0.0      # Resting release probability for facilitating synapse

# Initialize arrays
pre_spikes = np.zeros(n_steps)       # Presynaptic spike train (0 or 1)
P_depressing = np.ones(n_steps)      # P for depressing synapse
P_facilitating = np.zeros(n_steps)   # P for facilitating synapse
trans_depressing = np.zeros(n_steps) # Transmissions from depressing synapse
trans_facilitating = np.zeros(n_steps) # Transmissions from facilitating synapse

# Set initial values
P_depressing[0] = P0_depressing
P_facilitating[0] = P0_facilitating

# Generate presynaptic spike train and update synapse states
for i in range(1, n_steps):
    # Generate a spike with probability r*dt (convert dt to seconds)
    spike_probability = r * dt_sec
    
    if np.random.rand() < spike_probability:
        # Record presynaptic spike
        pre_spikes[i] = 1
        
        # For depressing synapse: check for transmitter release
        if np.random.rand() < P_depressing[i-1]:
            trans_depressing[i] = 1
            P_depressing[i] = 0  # P resets to 0 after release
        else:
            # No transmission, P remains unchanged for this step
            P_depressing[i] = P_depressing[i-1]
            
        # For facilitating synapse: check for transmitter release
        if np.random.rand() < P_facilitating[i-1]:
            trans_facilitating[i] = 1
            
        # P increases by 0.1(1-P) for facilitating synapse regardless of transmission
        P_facilitating[i] = P_facilitating[i-1] + 0.1 * (1 - P_facilitating[i-1])
        
    else:
        # No presynaptic spike
        # Update P values according to differential equations
        
        # For depressing synapse: τ_P * dP/dt = P0 - P
        P_depressing[i] = P_depressing[i-1] + dt/tau_P_depressing * (P0_depressing - P_depressing[i-1])
        
        # For facilitating synapse: τ_P * dP/dt = P0 - P
        P_facilitating[i] = P_facilitating[i-1] + dt/tau_P_facilitating * (P0_facilitating - P_facilitating[i-1])

# Create stacked plots
plt.figure(figsize=(12, 10))

# Plot 1: Presynaptic spikes
plt.subplot(5, 1, 1)
plt.vlines(time[pre_spikes > 0], 0, 1, color='k')
plt.ylabel('Presynaptic\nspikes')
plt.title(f'Simulation with Firing Rate r = {r} Hz')
plt.xlim(0, T)

# Plot 2: Depressing synapse transmissions
plt.subplot(5, 1, 2)
plt.vlines(time[trans_depressing > 0], 0, 1, color='r')
plt.ylabel('Depressing\ntransmissions')
plt.xlim(0, T)

# Plot 3: Facilitating synapse transmissions
plt.subplot(5, 1, 3)
plt.vlines(time[trans_facilitating > 0], 0, 1, color='g')
plt.ylabel('Facilitating\ntransmissions')
plt.xlim(0, T)

# Plot 4: P for depressing synapse
plt.subplot(5, 1, 4)
plt.plot(time, P_depressing)
plt.ylabel('P (depressing)')
plt.xlim(0, T)

# Plot 5: P for facilitating synapse
plt.subplot(5, 1, 5)
plt.plot(time, P_facilitating)
plt.xlabel('Time (ms)')
plt.ylabel('P (facilitating)')
plt.xlim(0, T)

plt.tight_layout()
plt.savefig("synapse_simulation.png", dpi=300)
plt.show()
