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
#         zi - Initial state of the filter sections. "state" as used in state-space model.
#              This is not the buffer values in DF1, DF2 etc.  
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

# ----------------------------------------------------------------------------------------
# is_sos_stable(sos_fp, fracbits)
#   Checks if a given sos filter with fixed point coefficients is stable using Schur-Cohn
# stability criterion
#
# Inputs: sos_fp - sos coef array with fixed point coefficients
#         fracbits - No. of fractional bits in the coefficients
#
# Output: true/false - Indicating whether the filter is stable or not
# ----------------------------------------------------------------------------------------
def is_sos_stable(sos_fp, fracbits):
    fp_1 = 1 << fracbits # Just 1 in fixed point representatin
    for b0,b1,b2,a0,a1,a2 in sos_fp:
        if np.abs(a2) < fp_1:
            if np.abs(fp_1+a2) > np.abs(a1):
                continue
            else:
                return False
        else:
            return False
    
    return True 
            

# ----------------------------------------------------------------------------------------
# sosfilt_fp
#   Filtering using SOS sections in DF1. This the fixed point version of sosfilt, but this
# doesn't use state-space implementation like sosfilt does. 
#
# Inputs: sos_fp - sos coef array with elements in int64 format
#         x_fp - input samples in int64 format
#         buf_vals_i - Initial values of the DF1 buffers
#         xy_res - resolution of input/output (bits) (For simplicity of biquad implementation, 
#                  we assume that input and output have the same resolution)
#         coef_res - resolution of numerator/denominator polynomial coefficients (Again, for
#                    the sake of simpler implementation, we are making the assumption that
#                    numerator and denominator coefficients have the same resolution
#         mul_out_res - resolution of multiplier output (bits)
#         acc_width - Accumulator width in bits
#         final_shift - No. of bits by which adder output must be shifted before
#                       assigning the filter output value. If this is a positive value,
#                       then the result is shifted up. Otherwise, it is shifted down.
#
# Outputs: y_fp - Output of the filter in int64
#          buf_vals_f - Final values of the DF1 buffer
#          sat_flag - Saturation flag. Set to 1 if saturation occurred during accumulation
#                     or when accumulator output is assigned to the output
# ----------------------------------------------------------------------------------------
def sosfilt_fp(sos_fp, x_fp, buf_vals, xy_res, coef_res, mul_out_res, acc_width, final_shift):
    # Initialize return value(s)
    y_fp = np.zeros(shape = x_fp.shape).astype(np.int64)
    sat_count = 0

    # Make sure input parameters aren't an illegal combination
    if xy_res > 32 or coef_res > 32 or mul_out_res > 64 or acc_width > 64: 
        print('Error: Bit resolution of some input parameter out of acceptable values')
        sys.exit()

    mul_shift_down = xy_res + coef_res - mul_out_res    # Some constants which will be 
                                                        # useful in the loop, but we 
                                                        # don't want to compute them
                                                        # again and again.

    for sample_idx,x in enumerate(x_fp): # We proceed sample-by-sample
        x_biquad = x # For the first biquad, biquad input is same as the filter input
        for sosidx,(b0,b1,b2,a0,a1,a2) in enumerate(sos_fp):
            # Note that we are taking some liberty below because in an actual hardware, each    
            # multiplication will happen one at a time along with accumulation and accumulator    
            # could overflow any time. We instead do the who MAC operation in one shot and 
            # saturate at end.                           
            acc = ((b0 * x_biquad) >> mul_shift_down) + \
                  ((b1 * buf_vals[sosidx][0]) >> mul_shift_down) + \
                  ((b2 * buf_vals[sosidx][1]) >> mul_shift_down) - \
                  ((a1 * buf_vals[sosidx][2]) >> mul_shift_down) - \
                  ((a2 * buf_vals[sosidx][3]) >> mul_shift_down)     
            
            if np.abs(acc) > (2**(acc_width-1)):                
                sat_count = sat_count + 1
                y_biquad = (2**(xy_res-1))-1 if acc > 0 else -1*(2**(xy_res-1))   # Max out the output
            else:
                temp_out = acc << final_shift if final_shift > 0 else acc >> final_shift
                if np.abs(temp_out) > (2**(xy_res-1)):                
                    sat_count = sat_count + 1
                    y_biquad = (2**(xy_res-1))-1 if temp_out > 0 else -1*(2**(xy_res-1))   # Max out the output
                else:
                    y_biquad = temp_out
             
            # Shift values in registers so that we will have updated buffers when we process
            # the  input sample. Note that, in an actual hardware, all buffer values of all
            # biquads will be updated simultaneously. We are doing it one sos at a time for 
            # algorithmic simplicity. Both give same results. For the purpose of this code,
            # it is OK to do this
            buf_vals[sosidx][1] = buf_vals[sosidx][0]
            buf_vals[sosidx][0] = x_biquad
            buf_vals[sosidx][3] = buf_vals[sosidx][2]
            buf_vals[sosidx][2] = y_biquad
            
            x_biquad = y_biquad # For the next sos, the output of this sos is its input                     

        y_fp[sample_idx] = y_biquad # Last sos output is also the overall output of the filter

    # Out of the main loop.         
    return y_fp, buf_vals, sat_count      
# ---------------------------------- Main code starts here ---------------------------------------------
# Filter paramters
cutoff_freq = np.pi/4   # radians. Angular frequency
order = 8 # no unit. IIR filter order. Keep it an even number for simplicity
n_freqsamples = 128 # no unit. 

# Parameters related to fixed point operations
coef_res = 32 # bits
coef_int_bits = 2 # bits. This is without considering the sign bit
xy_res = 32 # bits
xy_int_bits = 0 # bits. 
mul_out_res = 32 # bits. This is the output of the multiplier that goes into the accumulator
acc_width = 40 # bits. Width of the accumulator. 
final_shift = 3 # bits

# Design filter 
sos = signal.butter(btype='lowpass', N=order, Wn=cutoff_freq/np.pi, output='sos')
sos_fp = (np.round(sos*(2**(coef_res-coef_int_bits-1)))).astype(np.int64) # We will use this for the fixed point version
if not is_sos_stable(sos_fp, coef_res-coef_int_bits-1): # Proprietary function to check filter stability after the 
                                                        # coefficicents were converted to fixed point in the previous line.
                                                        # The possibility of the rounding operation rendering a stable filter
                                                        # unstable is extremely rare, but it is a good idea to make it a habit
                                                        # to always check for filter stability anytime we mess with its
                                                        # coefficients. Furthermore, I didn't find a python library function 
                                                        # that checks IIR filter stability of second order sections and took 
                                                        # the opportunity to implement one
    print('Fixed point IIR is unstable')
    system.exit()
                                                                      

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
 
dt = {1:np.int8, 2:np.int16, 4:np.int32}.get(wf.getsampwidth())
if dt == None:
    print('Unsupported input data resolution')
    sys.exit(0)


# We want to normalize the input signal to have a max value of 1 for two reasons:
#   1. It is easier to judge how big the output, intermediate stages in the biquad got by visual 
#      inspection if the input max is 1
#   2. We want to nake sure the input covers max dynamic range possible as we are trying to test
#      how big different stages in the IIR biquad sections can get
xmin, xmax = getminmax() # Function calculates the min, max (signed) values of the input data
xscale = max(np.abs(xmin), np.abs(xmax)) + 1    # The plus 1 here is to make sure we normalise the input 
                                                # to have values strictly less than 1. Typically xscale
                                                # value will be in thousands. So adding a 1 doesn't change
                                                # the value a lot. But it ensure max value of input remains
                                                # strictly below 1
zi = signal.sosfilt_zi(sos) # Get the initial state based on unit input signal's steady state response
ymax_vals_old = np.empty(shape = (sos.shape[0], ))
quant_error = []
sat_count_array = []
frate = wf.getframerate()
total_seconds = wf.getnframes()//frate+1 # Total number of seconds in the data
print('About to process', total_seconds, 'seconds of data')
for findex in range(total_seconds): # Process 1 sec of data at a time
                                    # to prevent overuse of RAM. 
    wf.setpos(frate*findex)
    rawdata = wf.readframes(frate) # Read 1 sec worth of samples
    temp = np.frombuffer(rawdata, dtype=dt) # Converting bytes object to np array
    xsamples = temp.reshape((wf.getnchannels(),-1), order='F')[0]/(xscale*8)    # Converting bytes object to
                                                                                # np array and normalising it 
    y, zi, ymax_vals = sosfilt(sos, xsamples, zi) # Output state zf returned is assigned to zi so that
                                                  # it will automatically become input state for the 
                                                  # next set of samples 
    ymax_vals_old = np.maximum(ymax_vals, ymax_vals_old) # Keep updating the max values of sos
                                                         # section outputs   

    # Sending same input data through fixed point DF1 filter to calculate quantization noise
    x_fp = (np.round(xsamples*(2**(xy_res-xy_int_bits-1)))).astype(np.int64) # The minus 1 is to exclude the sign bit
    buf_vals = np.zeros([sos_fp.shape[0], 4]).astype(np.int64)
    y_fp, buf_vals, sat_count = sosfilt_fp(sos_fp = sos_fp, 
                                          x_fp = x_fp, 
                                          buf_vals = buf_vals, 
                                          xy_res=xy_res, 
                                          coef_res = coef_res,
                                          mul_out_res = mul_out_res,
                                          acc_width = acc_width,
                                          final_shift = final_shift)
    sat_count_array.append(sat_count)

    # Compare the full precision floating point output with fixed point output and calculate the quantization
    # noise
    # quant_error.append(np.mean((y[y.shape[0]//8:]-y_fp[y.shape[0]//8:]/(2**(31)))**2))    
    quant_error.append(np.mean((y-y_fp/(2**(31)))**2))    
    print('Processed',findex+1, 'seconds of data')
 
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

#fig2 = go.Figure()
#yscaled = y_fp/(2**31)
#xscaled = x_fp/(2**31)
#lastidx = frate 
#fig2.add_trace(go.Scatter(x=np.arange(0,lastidx), y=y[:lastidx], name = 'Floating point output'))
#fig2.add_trace(go.Scatter(x=np.arange(0,lastidx), y=yscaled[:lastidx], name='Fixed-point output'))
#fig2.show()

