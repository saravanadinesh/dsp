""" In this code, we look at how to avoid overflow in IIR filters. 
    We compare a fixed point IIR with a full resolution floating 
    point IIR to see whether a particular input value is successful
    in avoiding overflow given a set of register lengths for the 
    coefficients, input and the multiplier. The code hardcodes the 
    filter to a Butterworth IIR LPF. But one can use this code for
    any filter of their choice """

import plotly.graph_objects as go 
from plotly.subplots import make_subplots

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_defer_js_import as dji

import numpy as np
from scipy import signal 
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics

import os
import sys
import wave
from tqdm import tqdm, trange

# Declare globals
wf = None
params = { 'coef_res':32, 'xy_res':32, 'mul_out_res':32, 'scale_factor':None}
run_clicks = 0

# Filter paramters
cutoff_freq = np.pi/4   # radians. Angular frequency
order = 8 # no unit. IIR filter order. Keep it an even number for simplicity
n_freqsamples = 128 # no unit. 

# Design filter 
sos = signal.butter(btype='lowpass', N=order, Wn=cutoff_freq/np.pi, output='sos')
zpk = signal.butter(btype='lowpass', N=order, Wn=cutoff_freq/np.pi, output='zpk')   # Just for the purpose of displaying
                                                                                    # pz plot

# Find time domain impulse response of the system
# Note: We have to do some monkeying around to find it mainly because the signal library's
# dimpulse function unnecessarily overcomplicates finding impulse response of a system. 
zpkl = list(zpk)
zpkl.append(1)
zpkt = tuple(zpkl)
hn_samples = 200 # This is how many samples to display to the user
hn_times, hn_temp = signal.dimpulse(system=zpkt, n = 200) 
hn = hn_temp[0].flatten() 

# Find out an h[n] based scale factor
params['scale_factor'] = np.sum(np.abs(hn[:50])) # 50 samples is the default. User can change later
hn_min = np.min(hn)
hn_max = np.max(hn)
hn_range = [hn_min-0.1, hn_max+0.1]

hnfig =     {'data':[go.Scatter(x=np.arange(0,200), y=hn),
                     go.Scatter(x=np.arange(0,50),  y=hn[:50] ),
                     go.Scatter(x=[100], y=[hn_range[1]-0.1], 
                                text='sum |h[n]| = {}'.format(params['scale_factor']), 
                                mode='text', textfont_size=14)],
              'layout': go.Layout(
                            title={'text': 'Filter impulse response', 'font': {'color': 'white'}, 'x': 0.5},
                            xaxis_title = 'n (samples)',
                            yaxis_title = 'h[n]',
                            autosize=True,
                            colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                            template='plotly_dark',
                            paper_bgcolor='rgba(0, 0, 0, 0)',
                            plot_bgcolor='rgba(0, 0, 0, 0)',
                            margin={'b': 75},
                            xaxis={'range':[0,200]},
                            yaxis={'range':hn_range}
                        )
            }

errfig =    {'data':[],
              'layout': go.Layout(
                            title={'text': 'Mean square error distribution', 'font': {'color': 'white'}, 'x': 0.5},
                            xaxis_title = 'MSE (no unit)',
                            yaxis_title = 'Relative frequency',
                            autosize=True,
                            colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                            template='plotly_dark',
                            paper_bgcolor='rgba(0, 0, 0, 0)',
                            plot_bgcolor='rgba(0, 0, 0, 0)',
                            margin={'b': 75},
                        )
            }



# App layout - This is just for the UI. The main code that does all the signal processing is all the way at the end in 
# def run(n_clicks) function. 
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True
mathjax_script = dji.Import(src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS-MML_SVG")
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            <script type="text/x-mathjax-config">
            MathJax.Hub.Config({
                tex2jax: {
                inlineMath: [ ['$','$'],],
                processEscapes: true
                }
            });
            </script>
            {%renderer%}
        </footer>
    </body>
</html>
"""
app.layout = html.Div(
    children=[
        html.Div(
            className='row',
            children=[
                html.Div(className='three columns div-user-controls',   # This is the left block of the app
                         children=[
                            html.Div(
                                style={'height':'20%'},
                                children=[
                                    html.H5("IIR filter analysis"),
                                    html.P("Type in wav filename"), 
                                    dcc.Input(id="filename", type="text"),
                                    html.Span(id="path-validity"),
                                    html.Div(id="filename-output")
                                ]
                            ), 
                            html.Div(
                                style={'height':'12%'},
                                children=[
                                    html.P("Coefficient bit resolution (8 to 32)"), 
                                    dcc.Input(id="coef-res", type="number", min=8, max=32, step=1, value=params['coef_res']),
                                ]
                            ), 
                            html.Div(
                                style={'height':'12%'},
                                children=[
                                    html.P("Filter input bit resolution (8 to 32)"), 
                                    dcc.Input(id="xy-res", type="number", min=8, max=32, step=1, value=params['xy_res']),
                                ]
                            ), 
                            html.Div(
                                style={'height':'12%'},
                                children=[
                                    html.P("Multiplier bit resolution (8 to 32)"), 
                                    dcc.Input(id="mul-out-res", type="number", min=8, max=60, step=1, value=params['mul_out_res']),
                                ]
                            ), 
                            html.Div(
                                className='div-for-radio',
                                style={'height':'15%'},
                                children=[
                                   html.P("Input scaling method:", style={"padding-top":"10px"}),
                                   dcc.RadioItems(
                                        id='scaling-method',
                                        options=[
                                                {'label':'Use $\sum |h[n]|$', 'value':'hn'},
                                                {'label':'Use custom', 'value': 'custom'}
                                        ],
                                        value = 'hn'
                                   ),
                                ]
                            ),
                            html.Div(
                                style={'height':'10%'},
                                children=[
                                    html.P("Custom scale factor"),
                                    dcc.Input(id="custom-scale", type="number"),
                                ]
                            ), 
                            html.Button('Run', id='run', n_clicks=0, disabled=False, style={'color': '#FF4F00', 'margin-top':'20px'}),
                            html.P("After hitting run, progress can be seen in python prompt", style={"padding-top":"5px"})
                         ],
                ),
                html.Div(className='nine columns div-for-charts bg-grey',   # This is the right block of the app
                         children=[
                            dcc.Graph(id='hn-graph', config={'displayModeBar': False}, style={'height':'45%'}, figure=hnfig),
                            html.Div(
                                children=[
                                    html.Div(
                                        className='div-for-slider',
                                        children=[
                                            dcc.Slider( id = "hn-slider", min=0, max=200, step=1, value=50)
                                        ]   
                                    ),
                                ],
                                style={'height':'6%'}
                            ),
                            dcc.Graph(id='err-graph', config={'displayModeBar': False}, style={'height':'45%'}, figure=errfig)
                         ]
                )
            ]
        ),
        mathjax_script
    ]
)


# -----------------------------  Function definitions -------------------------------------------------

# Main callback function. Most inputs, all outtputs have been lumped up into this function
@app.callback(Output('hn-graph', 'figure'),
                Output('filename-output', 'children'), 
                Output('path-validity', 'children'), 
                Output('path-validity', 'style'),
                [Input('filename', 'value'), 
                Input('filename','n_submit'),
                Input('coef-res', 'value'),
                Input('xy-res', 'value'),
                Input('mul-out-res', 'value'),
                Input('scaling-method', 'value'),
                Input('custom-scale', 'value'),
                Input('hn-slider', 'value')])
def update_output(input_filename, submit_times, coef_res, xy_res, mul_out_res, scaling_method, custom_scale, hn_slider):
    global wf, params, hn, hn_range, hnfig 

    tick_mark= '\u2714',{'font-family': 'wingdings', 'color':'Green', 'font-size':'100%', 'padding-left':'30px'}
    cross_mark = '\u274C', {'font-family': 'wingdings', 'color':'Crimson', 'font-size':'70%', 'padding-left':'30px'}
    

    ctx = dash.callback_context # Since there are too many inputs, find which one actually triggered the callback
    triggered_input_id = ctx.triggered[0]['prop_id'].split('.')[0]

    
    if triggered_input_id == 'filename': # If filename is valid, call dsp task and plot the charts for the first time
        if submit_times == (update_output.submit_times+1):
            update_output.submit_times = update_output.submit_times + 1
            if os.path.exists(input_filename):
                if (wf != None):
                    wf.close()
                
                try:
                    wf = wave.open(input_filename, 'r')
                except Exception as e:
                    return hnfig, 'File could not be opened', *cross_mark
                
                return hnfig, '',*tick_mark
            else:
                return hnfig, 'File does not exist',*cross_mark                 

        
        if input_filename: # Show tick or cross marks to indicate to the user that they are typing in -
                           # a valid path 
            if input_filename[-1] == '/': # Everytime user enters '/' check if the directory is valid
                if os.path.exists(input_filename):
                    return hnfig, '', *tick_mark     
                else:
                    return hnfig, 'Incorrect path', *cross_mark           
        
        return hnfig, '',None, None
    
    # If other than filename field triggered callback call the dsp task and update the charts
    else:
        if triggered_input_id == 'coef-res':
            params['coef_res'] = coef_res 
        elif triggered_input_id == 'xy-res':
            params['xy_res'] = xy_res 
        elif triggered_input_id == 'mul-out-res':
            params['xy_res'] == mul_out_res 
        elif triggered_input_id == 'scaling-method':
            if scaling_method == 'hn':
                params['scale_factor'] = np.sum(np.abs(hn[:hn_slider])) 
            elif scaling_method == 'custom':
                params['scale_factor'] = custom_scale
        elif triggered_input_id == 'custom-scale':
            if scaling_method == 'custom':
                params['scale_factor'] = custom_scale 
        elif triggered_input_id == 'hn-slider':
            hn_scalefactor = np.sum(np.abs(hn[:hn_slider]))
            if scaling_method == 'hn':
                params['scale_factor'] = np.sum(np.abs(hn[:hn_slider])) 
            hnfig =     {'data':[go.Scatter(x=np.arange(0,200), y=hn),
                                 go.Scatter(x=np.arange(0,hn_slider),  y=hn[:hn_slider] ),
                                 go.Scatter(x=[100], y=[hn_range[1]-0.1], 
                                            text='sum |h[n]| = {}'.format(hn_scalefactor), 
                                            mode='text', textfont_size=14)],
                         'layout': go.Layout(
                                        title={'text': 'Filter impulse response', 'font': {'color': 'white'}, 'x': 0.5},
                                        xaxis_title = 'n (samples)',
                                        yaxis_title = 'h[n]',
                                        autosize=True,
                                        colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                                        template='plotly_dark',
                                        paper_bgcolor='rgba(0, 0, 0, 0)',
                                        plot_bgcolor='rgba(0, 0, 0, 0)',
                                        margin={'b': 75},
                                        xaxis={'range':[0,200]},
                                        yaxis={'range':hn_range}
                                    )
                        }
             
        return hnfig, '',None, None


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
            acc = ((b0 * x_biquad + (1 << (mul_shift_down-1))) >> mul_shift_down) + \
                  ((b1 * buf_vals[sosidx][0] + (1 << (mul_shift_down-1))) >> mul_shift_down) + \
                  ((b2 * buf_vals[sosidx][1] + (1 << (mul_shift_down-1))) >> mul_shift_down) - \
                  ((a1 * buf_vals[sosidx][2] + (1 << (mul_shift_down-1))) >> mul_shift_down) - \
                  ((a2 * buf_vals[sosidx][3] + (1 << (mul_shift_down-1))) >> mul_shift_down)     
            
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

@app.callback(Output('err-graph', 'figure'),
              Input('run', 'n_clicks'))
def run(n_clicks):
    global wf, params, sos, errfig

    if wf != None:
        coef_res =  params['coef_res'] 
        xy_res = params['xy_res'] 
        mul_out_res = params['mul_out_res'] 
        scale_factor = params['scale_factor']

        # ---------------------------------- Main code starts here ---------------------------------------------
        # Parameters related to fixed point operations
        
        coef_int_bits = np.floor(np.log2(np.max(np.abs(sos)))+1).astype(np.int64) # This is without considering the sign bit
        xy_int_bits = 0 # bits. We keep this as zero as there is no advantage to having integer bits while converting a integer. -
                        # But one wants to mess with it, one can. So I just wanted to provide a placeholder 
        acc_width = mul_out_res + 3 # bits. Width of the accumulator. One biquad involves accumulating 5 multiplications. So 3
                                    # extra bits than mul_out_res are sufficient
        xy_frac_bits = xy_res - xy_int_bits - 1 # Just makes understanding the code easier
        coef_frac_bits = coef_res - coef_int_bits - 1 # Just makes understanding the code easier
        mul_out_frac_bits = mul_out_res - (xy_int_bits + 1) - (coef_int_bits+1) # Suppose we multiply x*h, where x is Q31 number,
                                                                                # and h is a  Q2.29 number, then the result will be
                                                                                # a Q3.60 bits (and not QQ2.60 as some text books say). 
                                                                                # But if mul_out_res = 32, then we need to downshift 
                                                                                # Q3.60 by 32 bits, resulting in Q3.28. The formula here
                                                                                # is right because, 28 = 32 - (0+1) - (2+1)
        final_shift = xy_frac_bits - mul_out_frac_bits  # Acculator output will also have the same number of fractional bits as that of the
                                                        # multiplier. Continuing the example in the previous comment, this will be 28 bits. 
                                                        # Since we want the output to have the same resolution as the input, we want to 
                                                        # have 31 fractinal bits in it. That means shifting the accumulator output by 3,
                                                        # i.e., 31-28. This is what this line of the code achieves 
        
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
                                                                              
        
        
        dt = {1:np.int8, 2:np.int16, 4:np.int32}.get(wf.getsampwidth())
        if dt == None:
            print('Unsupported input data resolution')
            sys.exit(0)
        
        zi = signal.sosfilt_zi(sos)        
        ymax_vals_old = np.empty(shape = (sos.shape[0], ))
        quant_error = []
        sat_count_array = []
        frate = wf.getframerate()
        total_seconds = wf.getnframes()//frate+1 # Total number of seconds in the data
        normalizing_factor = 2**(8*wf.getsampwidth()-1) # The minus one is to leave out the sign bit
        for findex in tqdm(range(total_seconds), desc="Processing data"): # Process 1 sec of data at a time
                                            # to prevent overuse of RAM. 
            wf.setpos(frate*findex)
            rawdata = wf.readframes(frate) # Read 1 sec worth of samples
            temp = np.frombuffer(rawdata, dtype=dt) # Converting bytes object to np array
            xmono_int = temp.reshape((wf.getnchannels(),-1), order='F')[0]
            xsamples = xmono_int/(normalizing_factor*scale_factor)  # Converting bytes object to
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
            if np.mean(y) != 0: # Sometimes, at the end of audio files, we will zero data. In this case, the error will turn out to be zero -
                                # It is a good idea to avoid adding them to the quant_error list because it will make the histogram of 
                                # it less revealing of the error distribution
                error = (y - y_fp/(2**xy_frac_bits))
                relative_error = error/y
                argmax_val = np.argmax(np.abs(relative_error))
                print('Max error:', relative_error[argmax_val])
                print('y:', y[argmax_val])
                print('yp:', y_fp[argmax_val]/(2**xy_frac_bits)) 
                error_mean_var = [np.mean(relative_error), np.var(relative_error)]
                quant_error.append(error_mean_var)    
    
            print(error_mean_var)
            processed_seconds = findex+1
       
        print(sat_count_array)

        errfig = {'data':[go.Histogram(x=quant_error, histnorm='probability')],
                  'layout': go.Layout(
                                    title={'text': 'Mean square error distribution', 'font': {'color': 'white'}, 'x': 0.5},
                                    xaxis_title = 'MSE (no unit)',
                                    yaxis_title = 'Relative frequency',
                                    autosize=True,
                                    colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                                    template='plotly_dark',
                                    paper_bgcolor='rgba(0, 0, 0, 0)',
                                    plot_bgcolor='rgba(0, 0, 0, 0)',
                                    margin={'b': 75},
                                )
                    }

    
    return errfig   # This here hasn't got a purpose - Dash demands that we always return something!


#@app.callback(Output('run', 'disabled'),
#              Input('run', 'n_clicks'))
#def run_enable_disable(n_clicks):
#    global run_clicks
#    
#    print('n_clicks, run_clicks: ', n_clicks, run_clicks)
#    if n_clicks > run_clicks:
#        run_clicks = n_clicks
#        return True
#    else:
#        return False

update_output.submit_times = 0 # This variable should not be changed outside the update_output function

# Run the html app
if __name__ == '__main__':
    app.run_server(debug=True)


