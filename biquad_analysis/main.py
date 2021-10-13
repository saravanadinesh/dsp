""" In this code, we look at the max value the output of a biquad 
    can get to. The objective is to determine what should be the 
    bit-width of registers, adders, multipliers in the biquad. We
    will use a Butterworth IIR LPF to do this analysis """

import numpy as np
import plotly.graph_objects as go 
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import signal 
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics

# Filter paramters
Fs = 48000 # Hz. sampling frequency
cutoff_freq = np.pi/4   # radians. Angular frequency
order = 2 # no unit. IIR filter order. Keep it an even number for simplicity
n_freqsamples = 128
# Derived parameters
Ts = 1/Fs # seconds. Time difference between two consecutive samples

# Design filter 
sos = signal.butter(btype='lowpass', N=order, Wn=cutoff_freq/np.pi, output='sos')
zpk = signal.butter(btype='lowpass', N=order, Wn=cutoff_freq/np.pi, output='zpk')   # Just for the purpose of displaying
                                                                                    # pz plot
[w, H] = signal.freqz_zpk(z=zpk[0], p=zpk[1], k=zpk[2], worN=n_freqsamples)   # Get the frequency response (just for displaying)

# Find filter roll-off by fitting a line to the transition band in the 
# immediate vicinity of the cutoff frequency 
Hmag = 20*np.log10(np.abs(H))
start_fidx = int(cutoff_freq*n_freqsamples/np.pi)                               # line x axis starts from cutoff frequency
end_fidx = start_fidx+25 if (start_fidx+25) < n_freqsamples else n_freqsamples  # and ends 25 frequency samples later
log2Freqs = np.log2(w[start_fidx:end_fidx]) # Select frequencies of interest, convert them to octave

x = log2Freqs.reshape((-1,1))   # Line fitting using sklearn lin regression function
y = Hmag[start_fidx:end_fidx]
model = LinearRegression()
model.fit(x,y)
rolloff = model.coef_   # Slope of the fitted line is roll off in dB/8ve
p = model.predict(x)

# --------------------------- Visualisation ---------------------------------------
fig1 = make_subplots(rows=2,cols=1, specs=[[{'type':'polar'}],[{'type':'xy'}]],
                     subplot_titles=['Pole-zero plot', 'Frequency response'])

## pole-zero plot
fig1.add_trace(go.Scatterpolar( r = np.abs(zpk[0]), theta = np.angle(zpk[0])*(180/np.pi), 
                                mode='markers', marker=dict(symbol='circle', size=12), name='zeros'), 
               row=1, col=1)
fig1.add_trace(go.Scatterpolar( r = np.abs(zpk[1]), theta = np.angle(zpk[1])*(180/np.pi), 
                                mode='markers', marker=dict(symbol='x', size=12), name='poles'), 
               row=1, col=1)


## Transfer function
fig1.add_trace(go.Scatter(x=w, y=Hmag, name='Gain'),
               row=2, col=1)
fig1.add_trace(go.Scatter(x=w[start_fidx:end_fidx], y=p, name='Roll off'),
               row=2, col=1)
fig1.add_trace(go.Scatter(x=[2.5], y=[-10], text='rolloff = {} (dB/8ve)'.format(round(rolloff[0],2)), 
               mode='text', textfont_size=14),
               row=2, col=1)
fig1.update_layout({'xaxis':{'title':{'text':'Frequency (radians)'}}})
fig1.update_layout({'yaxis':{'title':{'text':'Magnitude (dB)'}}})

fig1.show()
