import numpy as np
import matplotlib.pyplot as plt


c_m = 0.01  
g_L = 0.003
g_K = 0.36
g_Na = 1.2

E_L = -54.387
E_K = -77
E_Na = 50

V_init = -65 
m_init = 0.0529
h_init = 0.5961
n_init = 0.3177

def alpha_n(V):
    return 0.01 * (V + 55) / (1 - np.exp(-0.1 * (V + 55)))

def beta_n(V):
    return 0.125 * np.exp(-0.0125 * (V + 65))

def n_inf(V):
    return alpha_n(V) / (alpha_n(V) + beta_n(V))

def tau_n(V):
    return 1 / (alpha_n(V) + beta_n(V))

def alpha_m(V):
    return 0.1 * (V + 40) / (1 - np.exp(-0.1 * (V + 40)))

def beta_m(V):
    return 4 * np.exp(-0.0556 * (V + 65))

def m_inf(V):
    return alpha_m(V) / (alpha_m(V) + beta_m(V))

def tau_m(V):
    return 1 / (alpha_m(V) + beta_m(V))

def alpha_h(V):
    return 0.07 * np.exp(-0.05 * (V + 65))

def beta_h(V):
    return 1 / (1 + np.exp(-0.1 * (V + 35)))

def h_inf(V):
    return alpha_h(V) / (alpha_h(V) + beta_h(V))

def tau_h(V):
    return 1 / (alpha_h(V) + beta_h(V))

dt = 0.01 
T = 100
time = np.arange(0, T, dt)

V = np.zeros(len(time))
m = np.zeros(len(time))
h = np.zeros(len(time))
n = np.zeros(len(time))

V[0] = V_init
m[0] = m_init
h[0] = h_init
n[0] = n_init


I_e = np.zeros(len(time))
I_e[0:500] = -0.05  


for i in range(1, len(time)):
    I_Na = g_Na * (m[i-1]**3) * h[i-1] * (V[i-1] - E_Na)
    I_K = g_K * (n[i-1]**4) * (V[i-1] - E_K)
    I_L = g_L * (V[i-1] - E_L)
    I_ion = I_Na + I_K + I_L

    dV_dt = (-I_ion + I_e[i]) / c_m
    V[i] = V[i-1] + dV_dt * dt
    
    dn_dt = (n_inf(V[i-1]) - n[i-1]) / tau_n(V[i-1])
    dm_dt = (m_inf(V[i-1]) - m[i-1]) / tau_m(V[i-1])
    dh_dt = (h_inf(V[i-1]) - h[i-1]) / tau_h(V[i-1])
    
    n[i] = n[i-1] + dn_dt * dt
    m[i] = m[i-1] + dm_dt * dt
    h[i] = h[i-1] + dh_dt * dt


plt.figure(figsize=(10, 8))
plt.plot(time, V)
plt.ylabel('V (mV)')
plt.xlabel('Time (ms)')
plt.title('Hodgkin-Huxley Model')

plt.figure(figsize=(10, 4))
plt.plot(time, I_e)
plt.ylabel('Ie/A(nA/mm2)')
plt.xlabel('Time (ms)')
plt.title('Injected Current')
plt.show()

