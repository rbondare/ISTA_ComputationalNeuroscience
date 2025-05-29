import numpy as np
import matplotlib.pyplot as plt

delta_t = 0.001 # in seconds
time = np.arange(0, 1, delta_t)  

r = np.zeros_like(time)
r[(time >= 0.5) & (time <= 0.6)] = 100
P = r * delta_t 

spike_train = np.random.rand(len(time)) < P

print(np.unique(spike_train, return_counts=True))

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

        # facilitating synapse
        P_facilitating[i] += 0.1 * (1 - P_facilitating[i])
        if np.random.rand() < P_facilitating[i]:
            facilitating_synapse[i] = 1



    
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time, spike_train, label='Presynaptic Spike Train', color='black')
plt.ylabel('Spike Train')
plt.title('Presynaptic Spike Train and Synaptic Transmission')
plt.subplot(3, 1, 2)
plt.plot(time, depressing_synapse, label='Depressing Synapse', color='red')
plt.plot(time, P_depressing, label='Probability depressing', color='orange', linestyle='--')
plt.legend()
plt.ylabel('Depressing Synapse')
plt.subplot(3, 1, 3)
plt.plot(time, facilitating_synapse, label='Facilitating Synapse', color='green')
plt.plot(time, P_facilitating, label='Probability acilitating', color='blue', linestyle='--')
plt.legend()
plt.ylabel('Facilitating Synapse')
plt.xlabel('Time (s)')
plt.tight_layout()
#plt.savefig('synaptic_transmission_100Hz_burst.png')
plt.show()    

    
    




