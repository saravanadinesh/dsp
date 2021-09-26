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

temp_tdfig = {'data':[],
              'layout': go.Layout(
                            title={'text': 'Filter transfer function', 'font': {'color': 'white'}, 'x': 0.5},
                            xaxis_title = 'freq (Hz)',
                            yaxis_title = 'Magnitude (dBFS)',
                            autosize=True,
                            colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                            template='plotly_dark',
                            paper_bgcolor='rgba(0, 0, 0, 0)',
                            plot_bgcolor='rgba(0, 0, 0, 0)',
                            margin={'b': 75},
                            xaxis={'range':[0,24000]},
                            yaxis={'range':[-500, 0]}
                        )
            }
temp_fdfig = {'data':[],
              'layout': go.Layout(
                            title={'text': 'Input,output spectrums', 'font': {'color': 'white'}, 'x': 0.5},
                            xaxis_title = 'freq (Hz)',
                            yaxis_title = 'Magnitude (dBFS)',
                            autosize=True,
                            colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                            template='plotly_dark',
                            paper_bgcolor='rgba(0, 0, 0, 0)',
                            plot_bgcolor='rgba(0, 0, 0, 0)',
                            margin={'b': 75},
                            xaxis={'range':[0,24000]},
                            yaxis={'range':[-500, 0]}
                        )
            }

# App layout
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True
#"#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'
headings_col = "#FF0056"
app.layout = html.Div(
    children=[
        html.Div(
            className='row',
            children=[
                html.Div(className='three columns div-user-controls',   # This is the left block of the app
                         children=[
                            html.Div(
                                style={'height':'5%'},
                                children=[
                                    html.H5("FIR THDN tool"),
                                ]
                            ), 
                            html.Div(
                                className='div-for-radio',
                                style={'height':'11%'},
                                children=[
                                   html.P("Sampling Freqency Fs:", style={"color": headings_col, "padding-top":"10px"}),
                                   dcc.RadioItems(
                                        id='sampling-freq',
                                        options=[
                                                {'label':'48 KHz', 'value':'48'},
                                                {'label':'44.1 KHz', 'value':'44'}
                                        ],
                                        value='48',
                                        labelStyle={'display': 'inline-block'}
                                   )
                                ]
                            ),
                            html.Div(
                                className='div-for-slider',
                                style={'height':'11%'},
                                children=[
                                   html.P("Cutoff frequency fc:", 
                                          style={"color": headings_col, 'display':'inline-block',"padding-top":"10px"}
                                   ),
                                   html.Div(id='cutoff-freq-text', 
                                            style={'display':'inline-block'},
                                            children=html.P('11 KHz')
                                   ),
                                   dcc.Slider( id = "cutoff-freq", min=1, max=23, step=1, value=12),
                                ]
                            ),
                            html.Div(
                                className='div-for-slider',
                                style={'height':'11%'},
                                children=[
                                   html.P("Input frequency fc:", 
                                          style={"color": headings_col, 'display':'inline-block',"padding-top":"10px"}
                                   ),
                                   html.Div(id='input-freq-text', 
                                            style={'display':'inline-block'},
                                            children=html.P('4 KHz')
                                   ),
                                   dcc.Slider( id = "input-freq", min=1, max=23, step=1, value=4),
                                ]
                            ),
                            html.Div(
                                className='div-for-slider',
                                style={'height':'11%'},
                                children=[
                                   html.P("Coefficient resolution:", 
                                          style={"color": headings_col, 'display':'inline-block',"padding-top":"10px"}
                                   ),
                                   html.Div(id='coef-res-text', 
                                            style={'display':'inline-block'},
                                            children=html.P('16 bits')
                                   ),
                                   dcc.Slider( id = "coef-res", min=8, max=32, step=1, value=16),
                                ]
                            ),
                            html.Div(
                                className='div-for-slider',
                                style={'height':'11%'},
                                children=[
                                   html.P("No of filter taps:", 
                                          style={"color": headings_col, 'display':'inline-block',"padding-top":"10px"}
                                   ),
                                   html.Div(id='n-taps-text', 
                                            style={'display':'inline-block'},
                                            children=html.P('31')
                                   ),
                                   dcc.Slider( id = "n-taps", min=1, max=101, step=1, value=31),
                                ]
                            ),
                            html.Div(
                                className='div-for-radio',
                                style={'height':'11%'},
                                children=[
                                   html.P("Input resolution:", style={"color": headings_col, "padding-top":"10px"}),
                                   dcc.RadioItems( 
                                        id = "input-res",
                                        options = [{'label':'16 bits', 'value':'16'},
                                                   {'label':'24 bits', 'value':'24'},
                                                   {'label':'32 bits', 'value':'32'}],
                                        value='16',
                                        labelStyle={'display': 'inline-block'}
                                    )          
                                ]
                            ),
                            html.Div(
                                className='div-for-slider',
                                style={'height':'11%'},
                                children=[
                                   html.P("Multiplier o/p resolution:", 
                                          style={"color": headings_col, "display":"inline-block", "padding-top":"10px"}
                                   ),
                                   html.Div(id='mult-res-text', 
                                            style={'display':'inline-block'},
                                            children=html.P('16 bits')
                                   ),
                                   dcc.Slider( id = "mult-res", min=8, max=32, step=1, value=16),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Button('Execute', id='execute', n_clicks=0),
                                    html.P(children='Choose parameters and press execute', id='execute-text', style={"padding-top":"10px"})
                                ]
                            )
                         ],
                ),
                html.Div(className='nine columns div-for-charts bg-grey',   # This is the right block of the app
                         children=[
                            dcc.Graph(id='td-graph', config={'displayModeBar': False}, style={'height':'45%'}, figure=temp_tdfig),
                            dcc.Graph(id='fd-graph', config={'displayModeBar': False}, style={'height':'45%'}, figure=temp_fdfig)
                         ]
                )
            ]
        )
    ]
)

# dsp_task
# This is the core function that actually generates filter coefficients, 
# quantizes coefficients, input, carries out filtering etc.
def dsp_task(fs, cutoff, sin_freq, numtaps, x_res, h_res, mul_out_res):

    # Print Parameters (Just for testing. You can comment out.
    params = {  'fs':fs, 
                'cutoff':cutoff, 
                'sin_freq':sin_freq, 
                'numtaps':numtaps, 
                'x_res':x_res, 
                'h_res':h_res, 
                'mul_out_res':mul_out_res
             }
    
    print("Parameters updated:")
    print(params)
    
    window = 'hamming'
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
    h_fdomain = rfft(h_fp/(2**(h_res-1)), numtaps) # Returns just one side of the FFT 
    h_mag = np.where(2*(1/numtaps)*abs(h_fdomain) == 0,         # Finding magnitude in dB. Avoiding
                     -500,                                      # zeros, replacing with -500dB so that
                     20*np.log10(2*(1/numtaps)*abs(h_fdomain))) # log10 doesn't throw errors and graph
                                                                # comes out OK
    
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
    
    input_THDN_dB = round(10*np.log10(input_THDN), 2)
    output_THDN_dB = round(10*np.log10(output_THDN), 2)
   
    tdfig = {'data':[go.Scatter(x=np.abs(fftfreq(numtaps, 1/fs)[:numtaps//2+1]), 
                                y=h_mag, name="Filter response")
                    ], 
            'layout': go.Layout(
                        xaxis_title = 'Frequency (Hz)',
                        yaxis_title = 'Magnitude (dB)',
                        title='Filter response',
                        autosize=True,
                        colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                        template='plotly_dark',
                        paper_bgcolor='rgba(0, 0, 0, 0)',
                        plot_bgcolor='rgba(0, 0, 0, 0)',
                        margin={'b': 75},
                        hovermode='x',
                        yaxis={'range': [-200, 0]}
                      )
            }
 
    thdn_text_xpos = [18000] if sin_freq<12000 else [1000]
    input_thdn_str = 'Input THDN: {} dB'.format(input_THDN_dB)
    output_thdn_str = 'Output THDN: {} dB'.format(output_THDN_dB)
    fdfig = {'data':[go.Scatter(x = np.abs(fftfreq(N, 1/fs)[:N//2+1]), 
                                y = x_mag, name='Input spectrum'),
                     go.Scatter(x = np.abs(fftfreq(N//2, 1/fs)[:N//4+1]), 
                                y = y_mag, mode='markers', name='Output spectrum'),
                     #go.Scatter(x = thdn_text_xpos, 
                     #           y = [-100], 
                     #           text = input_thdn_str, 
                     #           mode='text')
                    ], 
            'layout': go.Layout(
                        xaxis_title = 'Frequency (Hz)',
                        yaxis_title = 'Magnitude (dB)',
                        title='Input, Output spectrum' + '. ' + input_thdn_str + '. ' + output_thdn_str,
                        autosize=True,
                        colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                        template='plotly_dark',
                        paper_bgcolor='rgba(0, 0, 0, 0)',
                        plot_bgcolor='rgba(0, 0, 0, 0)',
                        margin={'b': 75},
                        hovermode='x',
                      )
            }
    return tdfig, fdfig
    

# This is the call back for the Execute button. This is the main callback
@app.callback(  Output('execute-text','children'),
                Output('td-graph', 'figure'),
                Output('fd-graph', 'figure'),
                Input('execute', 'n_clicks'),
                State('sampling-freq','value'),
                State('cutoff-freq','value'),
                State('input-freq','value'),
                State('n-taps', 'value'),
                State('input-res','value'),
                State('coef-res','value'),
                State('mult-res','value')
             )
def update_output(n_clicks, fs_str, cutoff, sin_freq, numtaps, x_res_str, h_res, mul_out_res):
    global temp_tdfig, temp_fdfig

    fs = {'48':48000, '44':44100}.get(fs_str)
    x_res = {'16':16, '24':24, '32':32}.get(x_res_str)


    # Check for incorrect combinations of inputs
    if mul_out_res > (x_res+h_res-2): # To avoid negative shift values in the filtering for loop
        error_msg = 'Multiplier resolution cannot be greater than input resolution + coefficient resolution + 2'
        return error_msg, temp_tdfig, temp_fdfig

    td_fig, fd_fig = dsp_task(fs, cutoff*1000, sin_freq*1000, numtaps, x_res, h_res, mul_out_res)
    return 'OK', td_fig, fd_fig
  
# Callback to restrict cutoff frequency slider to value below user chose fs
@app.callback( Output('cutoff-freq', 'max'),
               Output('cutoff-freq', 'value'),
               Input('sampling-freq', 'value')
             )
def update_cutoff_slider(fs_str):
    maxval = {'48':23, '44':22}.get(fs_str)
    value = {'48':12, '44':11}.get(fs_str)
    return maxval, value      

# Following callbacks are just for updating user selected values in the text areas
# of their correspinding sliders
@app.callback( Output('cutoff-freq-text', 'children'),
               Input('cutoff-freq', 'value')
             )
def update_cutoff_freq_text(value):
    return html.P(str(value)+' KHz')

@app.callback( Output('coef-res-text', 'children'),
               Input('coef-res', 'value')
             )
def update_coef_res_text(value):
    return html.P(str(value)+' bits')

@app.callback( Output('input-freq-text', 'children'),
               Input('input-freq', 'value')
             )
def update_input_freq_text(value):
    return html.P(str(value)+' KHz')

@app.callback( Output('n-taps-text', 'children'),
               Input('n-taps', 'value')
             )
def update_ntaps_text(value):
    return html.P(str(value))

@app.callback( Output('mult-res-text', 'children'),
               Input('mult-res', 'value')
             )
def update_mult_res_text(value):
    return html.P(str(value) + ' bits')


# Run the html app
if __name__ == '__main__':
    app.run_server(debug=True)

