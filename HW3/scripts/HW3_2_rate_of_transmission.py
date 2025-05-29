import numpy as np
import matplotlib.pyplot as plt

delta_t = 0.001 # in seconds
time = np.arange(0, 10, delta_t)  

firing_rates = np.arange(0, 100, 10)

depressing_transmission_rates = []
facilitating_transmission_rates = []


depressing_synapse = np.zeros(len(time))
facilitating_synapse = np.zeros(len(time))

tau_depressing = 0.3 # in seconds
tau_facilitating = 0.1 # in seconds
P0_depressing = 1
P0_facilitating = 0

P_depressing = np.zeros(len(time))
P_depressing[0] = P0_depressing

P_facilitating = np.zeros(len(time))
P_facilitating[0] = P0_facilitating


for r in firing_rates:
    print(f"Simulating with firing rate r = {r} Hz")
    
    P = r * delta_t 
    spike_train = np.random.rand(len(time)) < P    
    # Initialize arrays for this simulation
    depressing_synapse = np.zeros((len(time)))
    facilitating_synapse = np.zeros(len(time))
    

    P_depressing = np.zeros(len(time))
    P_depressing[0] = P0_depressing
    
    P_facilitating = np.zeros(len(time))
    P_facilitating[0] = P0_facilitating


    for i in range(1, len(time)):

        delta_P_depressing = ((P0_depressing - P_depressing[i-1])/tau_depressing) * delta_t
        P_depressing[i] = P_depressing[i-1] + delta_P_depressing

        delta_P_facilitating = ((P0_facilitating - P_facilitating[i-1])/tau_facilitating) * delta_t
        P_facilitating[i] = P_facilitating[i-1] + delta_P_facilitating

        if spike_train[i]:

            # depressing synapse
            if np.random.rand() < P_depressing[i]: 
                depressing_synapse[i] = 1
                P_depressing[i] = 0

                
            P_facilitating[i] += 0.1 * (1 - P_facilitating[i])
            if np.random.rand() < P_facilitating[i]:
                facilitating_synapse[i] = 1

    
    depressing_count = np.sum(depressing_synapse)
    facilitating_count = np.sum(facilitating_synapse)
    depressing_rate = depressing_count / time[-1] 
    facilitating_rate = facilitating_count / time[-1]
    depressing_transmission_rates.append(depressing_rate)
    facilitating_transmission_rates.append(facilitating_rate)


# plotting the results
plt.figure(figsize=(8, 6))
plt.plot(firing_rates, depressing_transmission_rates, label='Depressing Synapse', marker='o', color = 'red')
plt.plot(firing_rates, facilitating_transmission_rates, label='Facilitating Synapse', marker='o', color = 'blue')
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Transmission Rate')
plt.title('Transmission Rates vs Firing Rate')
plt.legend()
plt.savefig('rate of transmission vs firing rate.png')
plt.show()    

    
    




