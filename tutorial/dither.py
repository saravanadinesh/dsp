""" dither.py - Plots pics useful for explaining dither """

import matplotlib.pyplot as plt
import numpy as np

fs = 48000
sin_freq = 1000
N = int((fs/1000)*10)  # A factor of 10,20,30 etc. is useful for Fs=44.1KHz or its multiples as is typical in 
                       # audio. Also, a min factor of 10 is required for calculating THD correctly while using
                       # odd sin_freq KHz value (Ex. 3000) as we 
n = np.arange(0,N)
x = 7*np.sin(2*np.pi*(sin_freq/fs)*n)

fig, ax = plt.subplots()
ax.plot(n, x, color='cornflowerblue')
ax.set_xlabel('time (ms)')
ax.set_ylabel('amplitude')
ax.vlines(x=n[np.arange(1,N,4)], ymin=0, ymax=x[np.arange(0,N,4)], color='lightsteelblue', linestyles='dashed') 
ax.vlines(x=n[np.arange(5,N,int(fs/sin_freq))], ymin=0, ymax=x[np.arange(5,N,int(fs/sin_freq))], color='red')
ax.hlines(y=x[5], xmin=0, xmax=437, color='red', linestyles='dashed')
ax.text(transform=ax.transAxes, x=0.02, y=0.78, s='4.25', color='red') 
plt.show()
