import numpy as np
import matplotlib.pyplot as plt

# Parameters
delta_t = 0.001  # Time step in seconds
T = 10.0  # Total simulation time in seconds (longer to get better statistics)
time = np.arange(0, T, delta_t)
n_steps = len(time)

# Synapse parameters
tau_depressing = 0.3  # Time constant for depressing synapse in seconds
tau_facilitating = 0.1  # Time constant for facilitating synapse in seconds
P0_depressing = 1.0  # Resting release probability for depressing synapse
P0_facilitating = 0.0  # Resting release probability for facilitating synapse

# Range of firing rates to test
firing_rates = np.arange(0, 110, 10)  # From 0 to 100 Hz in steps of 10 Hz

# Arrays to store results
depressing_transmission_rates = []
facilitating_transmission_rates = []

# For each firing rate
for r in firing_rates:
    print(f"Simulating with firing rate r = {r} Hz")
    
    # Generate spike train with constant rate r
    P = r * delta_t  # Probability of spike in each time step
    spike_train = np.random.rand(n_steps) < P
    
    # Initialize arrays for this simulation
    depressing_synapse = np.zeros(n_steps)
    facilitating_synapse = np.zeros(n_steps)
    
    P_depressing = np.zeros(n_steps)
    P_depressing[0] = P0_depressing
    
    P_facilitating = np.zeros(n_steps)
    P_facilitating[0] = P0_facilitating
    
    # Run the simulation
    for i in range(1, n_steps):
        # Update release probabilities using differential equations
        delta_P_depressing = ((P0_depressing - P_depressing[i-1])/tau_depressing) * delta_t
        P_depressing[i] = P_depressing[i-1] + delta_P_depressing
        
        delta_P_facilitating = ((P0_facilitating - P_facilitating[i-1])/tau_facilitating) * delta_t
        P_facilitating[i] = P_facilitating[i-1] + delta_P_facilitating
        
        if spike_train[i]:
            # For depressing synapse
            if np.random.rand() < P_depressing[i]:
                depressing_synapse[i] = 1
                P_depressing[i] = 0  # Reset to 0 after release
            
            # For facilitating synapse
            P_facilitating[i] += 0.1 * (1 - P_facilitating[i])
            if np.random.rand() < P_facilitating[i]:
                facilitating_synapse[i] = 1
    
    # Calculate transmission rates (count transmissions and divide by total time)
    depressing_count = np.sum(depressing_synapse)
    facilitating_count = np.sum(facilitating_synapse)
    
    depressing_rate = depressing_count / T
    facilitating_rate = facilitating_count / T
    
    depressing_transmission_rates.append(depressing_rate)
    facilitating_transmission_rates.append(facilitating_rate)
    
    # Optional: If we want to save plots for individual simulations
    if r in [0, 10, 50, 100]:  # Save plots for specific rates
        plt.figure(figsize=(12, 8))
        
        # Use a short time window for visualization (first 1 second)
        plot_time = time[time < 1.0]
        plot_indices = len(plot_time)
        
        # Plot presynaptic spike train
        plt.subplot(3, 1, 1)
        plt.plot(plot_time, spike_train[:plot_indices], 'k|', markersize=15)
        plt.ylabel('Presynaptic\nspikes')
        plt.title(f'Synaptic Transmission at {r} Hz Presynaptic Firing Rate')
        
        # Plot depressing synapse transmission
        plt.subplot(3, 1, 2)
        plt.plot(plot_time, depressing_synapse[:plot_indices], 'r|', markersize=15)
        plt.plot(plot_time, P_depressing[:plot_indices], 'orange', linestyle='--')
        plt.ylabel('Depressing\nsynapse')
        plt.legend(['Transmissions', 'P value'])
        
        # Plot facilitating synapse transmission
        plt.subplot(3, 1, 3)
        plt.plot(plot_time, facilitating_synapse[:plot_indices], 'g|', markersize=15)
        plt.plot(plot_time, P_facilitating[:plot_indices], 'blue', linestyle='--')
        plt.ylabel('Facilitating\nsynapse')
        plt.xlabel('Time (s)')
        plt.legend(['Transmissions', 'P value'])
        
        plt.tight_layout()
        plt.savefig(f'synaptic_transmission_{r}Hz.png')
        plt.close()

# Plot the relationship between presynaptic firing rate and transmission rate
plt.figure(figsize=(10, 6))
plt.plot(firing_rates, depressing_transmission_rates, 'ro-', label='Depressing Synapse')
plt.plot(firing_rates, facilitating_transmission_rates, 'go-', label='Facilitating Synapse')
plt.xlabel('Presynaptic Firing Rate (Hz)')
plt.ylabel('Transmission Rate (Hz)')
plt.title('Transmission Rate vs. Presynaptic Firing Rate')
plt.legend()
plt.grid(True)
plt.savefig('transmission_rate_vs_firing_rate.png')
plt.show()

# Print some statistics for the report
print("\nResults Summary:")
print("Firing Rate (Hz) | Depressing Trans. Rate (Hz) | Facilitating Trans. Rate (Hz)")
print("-" * 75)
for i, r in enumerate(firing_rates):
    print(f"{r:14.1f} | {depressing_transmission_rates[i]:28.2f} | {facilitating_transmission_rates[i]:29.2f}")
