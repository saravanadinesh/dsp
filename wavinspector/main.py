""" main.py - The entire project is contained in this python file """

import wave
import numpy as np
from scipy.signal import get_window
from scipy.fft import rfft, fftfreq
import sys
import errno
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

wf = None
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
old_figure = temp_tdfig, temp_fdfig

# We are good to get the html aoo setup
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

app.layout = html.Div(
    children=[
        html.Div(
            className='row',
            children=[
                html.Div(className='three columns div-user-controls',
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
                            )
                            
                         ],
                ),
                html.Div(className='nine columns div-for-charts bg-grey',
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

def getminmax():
    global wf
    
    ymin = 0
    ymax = 0
    for fs in range(wf.getnframes()//wf.getframerate()+1):
        wf.setpos(wf.getframerate()*fs)
        rawdata = wf.readframes(wf.getframerate()) # Read 1 sec worth of samples
        dt = {1:np.int8, 2:np.int16, 4:np.int32}.get(wf.getsampwidth())
        if dt == None: # We don't support other sample widths, Ex: 24-bit
            return 0,0
        temp = np.frombuffer(rawdata, dtype=dt)
        npdata = temp.reshape((wf.getnchannels(),-1), order='F') # If the wav fiel is stereo, then we will get a 2-row numpy array
        if wf.getnchannels() > 2:
            return 0,0
        else:
            ymin = {0:ymin, 1:np.min(npdata)}.get(ymin > np.min(npdata))
            ymax = {0:ymax, 1:np.max(npdata)}.get(ymax < np.max(npdata))

    return ymin, ymax
                
        
def dsp_task(channel_value, fft_window, frame_start, window_of_interest, ymin, ymax):
    global wf, old_figure
    
    if wf == None:
        return old_figure 

    
    F_S = wf.getframerate()
    T_S = 1/F_S
    num_frames = wf.getnframes()

    wf.setpos(round(num_frames*frame_start))
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
    subdata = npdata[channel, subdata_start:subdata_end+1]

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

    N = subdata.shape[0]
    fddata = rfft(subdata*get_window(fft_window, N))
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
    

    ctx = dash.callback_context
    triggered_input_id = ctx.triggered[0]['prop_id'].split('.')[0]

    
    if triggered_input_id == 'filename':
        if submit_times == (update_output.submit_times+1):
            update_output.submit_times = update_output.submit_times + 1
            if os.path.exists(input_filename):
                # Read wav file and convert the data to a numpy array
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

        
        if input_filename:
            if input_filename[-1] == '/': # Everytime user enters '/' check if the directory is valid
                if os.path.exists(input_filename):
                    return *old_figure, '', *tick_mark     
                else:
                    return *old_figure, 'Incorrect path', *cross_mark           
        
        return *old_figure, '',None, None

    else:
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
            


update_output.submit_times = 0 # This variable should not be changed outside the update_output function

# Run the html app
if __name__ == '__main__':
    app.run_server(debug=True)


