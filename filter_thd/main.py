""" main.py: Provides a utility for measuring THDN based on 
    coeficient bit resolution. An FIR low pass filter is 
    used to demonstrate the utitlity. 

    Note: The code uses 'x', 'y' and 'h' to denote  x, 
    y and hficients of the filter respectively. I 
    have done this so that it is easier to understand what
    is happening in the code as you read through it """

import numpy as np
from scipy.signal import firwin
from scipy.signal import convolve
from scipy.fft import rfft, fftfreq

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

# Parameters
numtaps = 31
fs = 48000 # Hz
cutoff = fs/4 # Hz
window = 'hamming'
h_res = 16 # bits. Should be less than 64
x_res = 16 # bits. For both x
sin_freq = 3000 # Hz
mul_out_res = 24 # bits.

# Linear phase LPF design using window method
h = firwin(numtaps=numtaps, cutoff=cutoff, window=window, fs=fs) # h is in float64

# Convert filter coefficients, h, to fixed point representation
# Note: 
# 1. FIR design by windpw method windows the ideal response: h_d[n] = (Wc/pi)*sinc(Wc/pi * n), 
# which results in hficients with absolute values that are always smaller than 1. This means,
# the hficients won't need any integer bits. We just need 1 sign bit and rest can be fractional
# bits.
# 2. We use a numpty int64 no matter the resolution that the user chooses. We simply keep any 
# extra most significant bits as unused
h_fp = (np.round(h*(2**(h_res-1)))).astype(np.int64) # The minus 1 is to exclude the sign bit

# Calculate input THD
# We want to calculate THD for sinusoids of frequencies 1KHz, 2KHz, 3KHz and so on. Their harmonics will
# also be in that series of frequencies. So we want FFT to sample the frequency response exactly on those
# frequency points. That means we need to use at least a fs/1000 point FFT. We can go for any integer
# multiple of that.  
N = int((fs/1000)*10)  # A factor of 10,20,30 etc. is useful for Fs=44.1KHz or its multiples as is typical in 
                       # audio. Also, a min factor of 10 is required for calculating THD correctly while using
                       # odd sin_freq KHz value (Ex. 3000) as we 
n = np.arange(0,N)
x = np.sin(2*np.pi*(sin_freq/fs)*n)
x_fp = (np.round(x*(2**(x_res-1)))).astype(np.int64) # The minus 1 is to exclude the sign bit
x_fdomain = rfft(x_fp/(2**(x_res-1)), N) # Returns just one side of the FFT 
x_mag = np.where(2*(1/N)*abs(x_fdomain) == 0,           # Finding magnitude in dB. Avoiding
                 -500,                                  # zeros, replacing with -500dB so that
                 20*np.log10(2*(1/N)*abs(x_fdomain)))   # log10 doesn't throw errors and graph
                                                        # comes out OK
                                                                                 
                                                                                
      
# Send the x signal through the filter
M = numtaps # For notational convenience
zeros_fp = np.zeros(M-1).astype(np.int64)
x_fp = np.append(x_fp, zeros_fp) # We are utilising python way accessing last element
                                 # with index of -1, last but one with an index of -2
                                 # and so on to fill zeroes for negative indices of the
                                 # x signal making our filter loop code simpler
y = np.zeros(N+M-1).astype(np.int64) # This is the y signal as well as the accumulator y
mul_shift_down = x_res + h_res - 1 - mul_out_res # The extra 1 is because we don't need 
                                                        # two sign bits at the multiplier y
for n in range(N+M-1):
    for k in range(M):
        mul_out = ( h_fp[k]*x_fp[n-k] + (1 << (mul_shift_down-1)) ) >> mul_shift_down # Round and shift down to required res
        y[n] = y[n] + mul_out.astype(np.int64) # Multiplying two int64 results in float64 in python!



begin_sample = int(10*(fs/sin_freq) + (M-1)//2)
end_sample = begin_sample + N//2
#begin_sample = int((M-1)//2)
#end_sample = begin_sample + N

total_samples = end_sample - begin_sample
y_fdomain = rfft(y[begin_sample:end_sample]/(2**(mul_out_res-1)))
y_mag = np.where(2*(1/total_samples)*abs(y_fdomain) == 0,           # Finding magnitude in dB. Avoiding
                 -500,                                              # zeros, replacing with -500dB so that
                 20*np.log10(2*(1/total_samples)*abs(y_fdomain)))   # log10 doesn't throw errors and graph
                                                                    # comes out OK

y_float = convolve(x, h)
y_float_fdomain = rfft(y_float[begin_sample:end_sample])
y_float_mag = np.where(2*(1/total_samples)*abs(y_float_fdomain) == 0,           # Finding magnitude in dB. Avoiding
                       -500,                                                    # zeros, replacing with -500dB so that
                       20*np.log10(2*(1/total_samples)*abs(y_float_fdomain)))   # log10 doesn't throw errors and graph
                                                                                # comes out OK

# Calculate THDN
x_pow = abs(x_fdomain)**2
input_THDN = (np.sum(x_pow) - x_pow[(sin_freq*N)//fs])/x_pow[(sin_freq*N)//fs]
y_pow = abs(y_fdomain)**2
output_THDN = (np.sum(y_pow) - y_pow[(sin_freq*N//2)//fs])/y_pow[(sin_freq*N//2)//fs]

print("Input THDN: ", 10*np.log10(input_THDN), ' dB')
print("Output THDN: ", 10*np.log10(output_THDN), ' dB')

# Visualisation
disp_N = int(1*(fs/sin_freq))
fig = make_subplots(rows=2, cols=1)
fig.append_trace(go.Scatter(x = np.linspace(start=0, stop=disp_N/fs, num=disp_N, endpoint=False), 
                            y = (x_fp/(2**(x_res-1)))[:disp_N], name='Input'), row=1, col=1)
fig.append_trace(go.Scatter(x = np.linspace(start=0, stop=disp_N/fs, num=disp_N, endpoint=False), 
                            y = y_float[(M-1)//2:disp_N+(M-1)//2], name='Output'), row=1, col=1)
fig.append_trace(go.Scatter(x = np.abs(fftfreq(N, 1/fs)[:N//2+1]), 
                            y = x_mag, name='Input THDN'),row=2, col=1)
fig.append_trace(go.Scatter(x = np.abs(fftfreq(N//2, 1/fs)[:N//4+1]), 
                            y = y_mag, mode='markers', name='Output Fixed Point'),row=2, col=1)
fig.show()



