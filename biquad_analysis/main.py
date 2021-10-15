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

import sys
import wave

# Declare globals
wf = None


# -----------------------------  Function definitions -------------------------------------------------
#------------------------------------------------------------------------------------
# getminmax (copied from wavinspector project.)
#   Find the audio signal's range so that we can have a fixed y-axis scaling as we 
# scroll through the audio data using the main slider
# -----------------------------------------------------------------------------------
def getminmax():
    global wf
    
    ymin = 0
    ymax = 0
    for fs in range(wf.getnframes()//wf.getframerate()+1): # Process 1 sec of data at a time
                                                           # to prevent overuse of RAM. fs stands
                                                           # for frame start, not sampling frequency :)
        wf.setpos(wf.getframerate()*fs)
        rawdata = wf.readframes(wf.getframerate()) # Read 1 sec worth of samples
        dt = {1:np.int8, 2:np.int16, 4:np.int32}.get(wf.getsampwidth())
        if dt == None: # We don't support other sample widths, Ex: 24-bit
            return 0,0
        temp = np.frombuffer(rawdata, dtype=dt) # Converting bytes object to np array
        npdata = temp.reshape((wf.getnchannels(),-1), order='F') # If the wav fiel is stereo, 
                                                                 # then we will get a 2-row numpy array
        if wf.getnchannels() > 2:
            return 0,0
        else:
            ymin = {0:ymin, 1:np.min(npdata)}.get(ymin > np.min(npdata))
            ymax = {0:ymax, 1:np.max(npdata)}.get(ymax < np.max(npdata))

    return ymin, ymax

# ----------------------------------------------------------------------------------------
# sosfilt
#   This is our version of IIR second order section filtering. We are using this so that
# we can check how large the outputs of every stage in a biquad gets, try out different
# input scaling methods and so on. We will use scipy signal's sosfilt function, but instead
# of supplying it will all the sos funcitons at once, we will supply one section at a time,
# allowing us to play with input scaling and also record the output max value of each stage 
#
# Our intention here is to know how many extra bits (as compared to input resolution) adders,
# multipliers need to have in every biquad to avoid overflow when we implement the filter in 
# fixed point hardware. For fixed point implementation, Direct Form I is best (in terms of 
# coefficient quantization and associated non-linearities).  We will not do actually quantize 
# coeffs though. The idea in this project is not to examine the effects of coefficient 
# quantisation - We just want to know the dynamic range of various intermediate values in 
# the DF1 computation
#
# Note: Although we are only finding out the max value of the output of every sos section,
# one can theoretically find the size of every multiplier, adder outputs in that sos section
# from the output max size and input max size.  
#
# Inputs: sos - second order sections array output by filter design lib functions of scipy
#               signal
#         x - the input signal
#         zi - Initial state of the filter sections. 
# Outputs: y - The output signal
#          
# ---------------------------------------------------------------------------------------- 
def sosfilt(sos, x, zi):
    ymax_vals = np.empty(shape = (sos.shape[0], ))
    zf = np.empty(shape=zi.shape)

    y = x
    num_sections = sos.shape[0]
    for section in range(num_sections):
        y, zf[section] = signal.sosfilt(sos = sos[section].reshape(1,6), 
                                        x=y, 
                                        zi = zi[section].reshape(1,2))
        ymax_vals[section] = np.max(np.abs(y))

    return y, zf, ymax_vals           



# ---------------------------------- Main code starts here ---------------------------------------------
# Filter paramters
Fs = 48000 # Hz. sampling frequency
cutoff_freq = np.pi/4   # radians. Angular frequency
order = 4 # no unit. IIR filter order. Keep it an even number for simplicity
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


# Send a test input through the input to see how large the output can get
input_filename = '/home/dinesh/Documents/ClassicalWavFiles/mozart_minuet.wav'
try:
    wf = wave.open(input_filename, 'r')
except Exception as e:
    print('Error: Input file could not be opened')
    sys.exit(0)
 
# TODO: Add checks for mono and sample width other than standard ones
dt = {1:np.int8, 2:np.int16, 4:np.int32}.get(wf.getsampwidth())

#wout = wave.open('output.wav', 'w')
#wout.setparams(wf.getparams())


# We want to normalize the input signal to have a max value of 1 for two reasons:
#   1. It is easier to judge how big the output, intermediate stages in the biquad got by visual 
#      inspection if the input max is 1
#   2. We wabt ti naje sure the input covers max dynamic range possible as we are trying to test
#      how big different stages in the IIR biquad sections can get
xmin, xmax = getminmax() # Function calculates the min, max (signed) values of the input data
xscale = max(np.abs(xmin), np.abs(xmax)) + 1    # The plus 1 here is to make sure we normalise the input 
                                                # to have values strictly less than 1. Typically xscale
                                                # value will be in thousands. So adding a 1 doesn't change
                                                # the value a lot. But it ensure max value of input remains
                                                # strictly below 1
zi = signal.sosfilt_zi(sos) # Get the initial state based on unit input signal's steady state response
ymax_vals_old = np.empty(shape = (sos.shape[0], ))
for findex in range(wf.getnframes()//wf.getframerate()+1): # Process 1 sec of data at a time
                                                           # to prevent overuse of RAM. 
    wf.setpos(wf.getframerate()*findex)
    rawdata = wf.readframes(wf.getframerate()) # Read 1 sec worth of samples
    temp = np.frombuffer(rawdata, dtype=dt) # Converting bytes object to np array
    xsamples = temp.reshape((wf.getnchannels(),-1), order='F')[0]/xscale    # Converting bytes object to np
                                                                            # array and normalising it 
    y, zi, ymax_vals = sosfilt(sos, xsamples, zi) # Output state zf returned is assigned to zi so that
                                                  # it will automatically become input state for the 
                                                  # next set of samples 
    ymax_vals_old = np.maximum(ymax_vals, ymax_vals_old) # Keep updating the max values of sos
                                                         # section outputs   

 
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
