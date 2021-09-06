""" main.py - The entire project is contained in this python file """

import wave
import numpy as np
from scipy.signal import get_window
from scipy.fft import rfft, fftfreq
import sys
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

wf = None
selected_section = np.array([])
temp_tdfig = {'data':[],
              'layout': go.Layout(
                            xaxis_title = 'time (sec)',
                            yaxis_title = 'Amplitude (no unit)',
                            autosize=True,
                            colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                            template='plotly_dark',
                            paper_bgcolor='rgba(0, 0, 0, 0)',
                            plot_bgcolor='rgba(0, 0, 0, 0)',
                            margin={'b': 75},
                            xaxis={'range':[0,1]},
                            yaxis={'range':[-128, 127]}
                        )
            }
temp_fdfig = {'data':[],
              'layout': go.Layout(
                            title={'text': 'Frequency response', 'font': {'color': 'white'}, 'x': 0.5},
                            xaxis_title = 'freq (Hz)',
                            yaxis_title = 'Magnitude (dBFS)',
                            autosize=True,
                            colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                            template='plotly_dark',
                            paper_bgcolor='rgba(0, 0, 0, 0)',
                            plot_bgcolor='rgba(0, 0, 0, 0)',
                            margin={'b': 75},
                            xaxis={'range':[0,16000]},
                            yaxis={'range':[-100, 0]}
                        )
            }
old_figure = temp_tdfig, temp_fdfig # We are doing this so that even before we have some chart to show,
                                    # we wan't to display empty charts with proper layout

# App layout
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

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
                                    html.H2("Wav File Inspector"),
                                    html.P("Type in wav filename"), 
                                    dcc.Input(id="filename", type="text"),
                                    html.Span(id="path-validity"),
                                    html.Div(id="filename-output")
                                ]
                            ), 
                            html.Div(
                                className='div-for-radio',
                                style={'height':'20%'},
                                children=[
                                   html.H4("Select channel:", style={"padding-top":"30px"}),
                                   dcc.RadioItems(
                                        id='channel',
                                        options=[
                                                {'label':'channel 1', 'value':'ch1'},
                                                {'label':'channel 2', 'value':'ch2'}
                                        ],
                                        value='ch1'
                                   )
                                ]
                            ),
                            html.Div(
                                className='div-for-radio',
                                style={'height':'25%'},
                                children=[
                                   html.H4("FFT window:", style={"padding-top":"30px"}),
                                   dcc.RadioItems(
                                        id='fft-window',
                                        options=[
                                                {'label':'None (Rectangular)', 'value':'boxcar'},
                                                {'label':'Hamming', 'value': 'hamming'},
                                                {'label':'Hann', 'value':'hann'}
                                        ],
                                        value='boxcar'
                                   )
                                ]
                            ),
                            html.Button('Play selected', id='play', n_clicks=0, style={'color': '#FF4F00', 'margin-top':'30px'})
                         ],
                ),
                html.Div(className='nine columns div-for-charts bg-grey',   # This is the right block of the app
                         children=[
                            dcc.Graph(id='td-graph', config={'displayModeBar': False}, style={'height':'45%'}, figure=temp_tdfig),
                            html.Div(
                                children=[
                                    html.Div(
                                        className='div-for-slider',
                                        children=[
                                            dcc.Slider( id = "main-slider", min=0, max=1, step=0.001, value=0)
                                        ]   
                                    ),
                                    html.Div(
                                        className='div-for-slider',
                                        children=[
                                            dcc.RangeSlider( id = "sub-slider", min=0, max=1, step=0.001, value=[0.4,0.6])
                                        ]   
                                    )
                                ],
                                style={'height':'6%'}
                            ),
                            dcc.Graph(id='fd-graph', config={'displayModeBar': False}, style={'height':'45%'}, figure=temp_fdfig)
                         ]
                )
            ]
        )
    ]
)


#------------------------------------------------------------------------------------
# getminmax
#   Find the audio signal's range so that we can have a fixed y-axis scaling as we 
# scroll through the audio data using the main slider
# -----------------------------------------------------------------------------------
def getminmax():
    global wf
    
    ymin = 0
    ymax = 0
    for fs in range(wf.getnframes()//wf.getframerate()+1): # Process 1 sec of data at a time
                                                           # to prevent overuse of RAM
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
                
        
#------------------------------------------------------------------------------------
# dsp_task
#   This is the core task for this app. It multiplies a window function with the data
# selected using the rangle slider (orange section). Then it takes FFT and calculates
# the magnitude specturm.  
# -----------------------------------------------------------------------------------
def dsp_task(channel_value, fft_window, frame_start, window_of_interest, ymin, ymax):
    global wf, selected_section, old_figure
    
    if wf == None:
        return old_figure 

    
    F_S = wf.getframerate()
    T_S = 1/F_S
    num_frames = wf.getnframes()

    wf.setpos(round(num_frames*frame_start)) # Use the main slider location as the -
                                             # starting point of data we fetch into
                                             # the RAM
    rawdata = wf.readframes(F_S) # Read 1 sec worth of samples
    dt = {1:np.int8, 2:np.int16, 4:np.int32}.get(wf.getsampwidth())
    if dt == None: # We don't support other sample widths, Ex: 24-bit
        return old_figure
    temp = np.frombuffer(rawdata, dtype=dt)
    npdata = temp.reshape((wf.getnchannels(),-1), order='F') # If the wav fiel is stereo, then we will get a 2-row numpy array
    if wf.getnchannels() == 1:
        channel = 0
    else:
        channel = {'ch1':0, 'ch2':1}.get(channel_value)
   
    if channel == None:
        return old_figure
    
    subdata_start = round(window_of_interest[0]*F_S)
    subdata_end = round(window_of_interest[1]*F_S)
    subdata = npdata[channel, subdata_start:subdata_end+1] # One can use some extra -
                                                           # maths and directly read
                                                           # the subdata of interest
                                                           # into the RAM. I have avoided
                                                           # this in the interest of 
                                                           # making the code easily 
                                                           # understandable
    selected_section = subdata

    xmin = frame_start*num_frames/F_S  
    xmax = xmin + 1 
    xdata = np.linspace(start=xmin, stop=xmax, num=npdata.shape[1])
    xsubdata =  xdata[subdata_start:subdata_end+1]
    tdfig = {'data':[go.Scatter(x=xdata, y=npdata[channel], mode='lines',name='channel {}'.format(channel+1)),
                    go.Scatter(x=xsubdata, y=subdata, mode='lines', name='subframes')], 
            'layout': go.Layout(
                        xaxis_title = 'time (sec)',
                        yaxis_title = 'Amplitude (no unit)',
                        autosize=True,
                        colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                        template='plotly_dark',
                        paper_bgcolor='rgba(0, 0, 0, 0)',
                        plot_bgcolor='rgba(0, 0, 0, 0)',
                        margin={'b': 75},
                        hovermode='x',
                        xaxis={'range': [xmin, xmax]},
                        yaxis={'range': [ymin, ymax]})
            }

    N = subdata.shape[0] # We use the exactly the same N-point FFT as the t-domain number of samples
    fddata = rfft(subdata*get_window(fft_window, N)) # Returns just one side of the FFT
    magresp = 20*np.log10(2*(1/N)*np.abs(fddata)/(2**(wf.getsampwidth()*8-1))) # dBFS
    freqs = np.abs(fftfreq(N, T_S)[:N//2+1]) 
    fdfig = {'data': [go.Scatter(x=freqs, y=magresp, mode='lines')],
             'layout': go.Layout(
                        title={'text': 'Frequency response', 'font': {'color': 'white'}, 'x': 0.5},
                        xaxis_title = 'freq (Hz)',
                        yaxis_title = 'Magnitude (dBFS)',
                        autosize=True,
                        colorway=['#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                        template='plotly_dark',
                        paper_bgcolor='rgba(0, 0, 0, 0)',
                        plot_bgcolor='rgba(0, 0, 0, 0)',
                        margin={'b': 75},
                        hovermode='x')
            }
    return tdfig, fdfig

# Main callback function. Most inputs, all outtputs have been lumped up into this function
@app.callback(Output('td-graph', 'figure'),
                Output('fd-graph', 'figure'),
                Output('filename-output', 'children'), 
                Output('path-validity', 'children'), 
                Output('path-validity', 'style'),
                [Input('filename', 'value'), 
                Input('filename','n_submit'),
                Input('channel', 'value'),
                Input('fft-window', 'value'),
                Input('main-slider', 'value'),
                Input('sub-slider', 'value')])
def update_output(input_filename, submit_times, channel_value, fft_window, frame_start, window_of_interest):
    global wf, old_figure

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
                    return *old_figure, 'File could not be opened', *cross_mark
                
                update_output.ymin, update_output.ymax = getminmax()     
                td_fig, fd_fig =  dsp_task( channel_value, 
                                            fft_window, 
                                            frame_start, 
                                            window_of_interest, 
                                            update_output.ymin, 
                                            update_output.ymax)
                old_figure = td_fig, fd_fig
                return td_fig, fd_fig, '',*tick_mark
            else:
                return *old_figure, 'File does not exist',*cross_mark                 

        
        if input_filename: # Show tick or cross marks to indicate to the user that they are typing in -
                           # a valid path 
            if input_filename[-1] == '/': # Everytime user enters '/' check if the directory is valid
                if os.path.exists(input_filename):
                    return *old_figure, '', *tick_mark     
                else:
                    return *old_figure, 'Incorrect path', *cross_mark           
        
        return *old_figure, '',None, None

    else: # If other than filename field triggered callback call the dsp task and update the charts
        if (wf != None):
            td_fig, fd_fig =  dsp_task( channel_value, 
                                        fft_window, 
                                        frame_start, 
                                        window_of_interest, 
                                        update_output.ymin, 
                                        update_output.ymax)
            old_figure = td_fig, fd_fig
            return td_fig, fd_fig, '',*tick_mark
        
        return *old_figure, '',None, None
            

# Begin: Comment out this section if you don't want to deal with portaudio, sounddevice etc 
# Callback for playing the selected audio data (orange part selected using range slider)
import sounddevice as sd
@app.callback(Output('play', 'n_clicks'), Input('play', 'n_clicks'))
def play_section(n_clicks):
    global wf, selected_section

    if wf != None and selected_section.size != 0:
        sd.play(selected_section, wf.getframerate(), device=sd.default.device[1])
    
    return n_clicks # This here hasn't got a purpose - Dash demands that we always return something!
# End: Comment out this section if you don't want to deal with portaudio, sounddevice etc 


update_output.submit_times = 0 # This variable should not be changed outside the update_output function

# Run the html app
if __name__ == '__main__':
    app.run_server(debug=True)


