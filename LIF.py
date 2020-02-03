#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Berberian
"""

import numpy as np
import matplotlib.pyplot as plt

#%% LIF function ==============================================================

def LIF(N, dt, tmin, tmax, I):
    
    g_L     = 0.05                                                              # membrane conductance (default: 0.05)
    C_m     = 1                                                                 # membrane capacitance (default: 1)
    E_L     = -70                                                               # equilibrium potential (default: -70)
    V_th    = -54                                                               # spike threshold (default: -54)
    V_reset = -90                                                               # reset value after spike (default: -90)
    V_peak  = 0                                                                 # peak membrane potential value to reach at spike (default: 0)

    time = np.arange(tmin, tmax + dt, dt)                                       # time increments (ms)
    T = len(time)                                                               # time length (ms)
    lastSTs = np.zeros(N)                                                       # preallocate last spike times
    STs = [ [] for x in range(N) ]                                              # create list(s) for storing spike times
    
    V = np.zeros(N) + E_L                                                       # preallocate membrane potential 
    Vtime = np.zeros([N, T])                                                    # preallocate membrane potential for storage
    
    s = 1                                                                       # initialize time
    while s-1 < T:
        dV = (g_L * (E_L - V) + I)/C_m                                          # computate membrane potential
        V = V + dV * dt                                                         # increment V by timestep dt via Euler method
        Vtime[:,s-1] = V                                                        # store membrane potential
        Spikers = [y for y,i in enumerate(V) if i > V_th]                       # find spikers greater than V_th
        if Spikers:                                                             # if there are spikers
            lastSTs[Spikers] = s-1                                              # store last spike times of spikers
            V[Spikers] = V_reset                                                # reset potential of spikers
            Vtime[:,s-1] = V_peak                                               # insert artificial spike up to 0
            [STs[i].append(s-1) for y,i in enumerate(Spikers)]                  # store post spike times
        s += 1                                                                  # time increment

    return [time, Vtime]

#%% specify LIF function arguments ============================================

N = 1                                                                           # specify the number of units
dt = 0.1                                                                        # specify timestep increments (ms)
tmin = 0                                                                        # specify min time (ms)
tmax = 1000                                                                     # specify max time (ms)
I = np.zeros(N) + 1                                                             # specify the amplitude of the injected input current (default: 1)

[time, Vtime] = LIF(N, dt, tmin, tmax, I)                                       # call LIF function

#%% plotting membrane potential V across time =================================

fig, ax = plt.subplots(figsize=(8,6))                                           # generate a figure
plt.plot(time,Vtime.transpose(), linewidth=2, color='k')                        # plot
plt.xlabel('Time (ms)',fontsize=14)                                             # specify xlabel
plt.ylabel('Membrane potential (mV)',fontsize=15)                               # specify ylabel
plt.xticks(fontsize=15)                                                         # specify xticks 
plt.yticks(fontsize=15)                                                         # specify yticks 
plt.xlim(tmin, tmax)                                                            # specify xlim
plt.show()                                                                      # show figure
