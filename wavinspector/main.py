""" main.py - The entire project is contained in this python file """

import wave
import numpy as np
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


# We are good to get the html aoo setup
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

app.layout = html.Div(
    children=[
        html.Div(
            className='row',
            children=[
                html.Div(className='four columns div-user-controls',
                         children=[
                            html.H2("Wav File Inspector"),
                            html.P("Type in wav filename"), 
                            dcc.Input(id="filename", type="text"),
                            html.Span(id="path-validity"),
                            html.Div(id="filename-output")
                         ],
                ),
                html.Div(className='eight columns div-for-charts bg-grey',
                         children=[
                             dcc.Graph(id='timeseries', config={'displayModeBar': False}, animate=True),
                             dcc.Graph(id='change', config={'displayModeBar': False})
                         ]
                )
            ]
        )
    ]
)


@app.callback(Output('filename','n_submit'), 
              Output('filename-output', 'children'), 
              Output('path-validity', 'children'), Output('path-validity', 'style'),
              [Input('filename', 'value'), Input('filename','n_submit')])
def update_output(input_filename, submit_times):
        global wf

        tick_mark= '\u2714',{'font-family': 'wingdings', 'color':'Green', 'font-size':'100%', 'padding-left':'30px'}
        cross_mark = '\u274C', {'font-family': 'wingdings', 'color':'Crimson', 'font-size':'70%', 'padding-left':'30px'}
        if submit_times:
            if os.path.exists(input_filename):
                # Read wav file and convert the data to a numpy array
                if (wf != None):
                    wf.close()
                try:
                    wf = wave.open(input_filename, 'r')
                except Exception as e:
                    return 0, 'File could not be opened', *cross_mark

                return 0,'Good',*tick_mark
            else:
                return 0,'File does not exist',*cross_mark                 
            
        
        if input_filename:
            if input_filename[-1] == '/': # Everytime user enters '/' check if the directory is valid
                if os.path.exists(input_filename):
                    return 0,'', *tick_mark     
                else:
                    return 0,'Incorrect path', *cross_mark           
        
        return 0,'',None, None


#temp1 = wf.readframes(-1)
#dt = {1:np.int8, 2:np.int16, 4:np.int32}.get(wf.getsampwidth())
#if not dt: # We don't support other sample widths, Ex: 24-bit
#    raise ValueError('Bit width not supported. Bytes'.format(wf.getsampwidth()))
#
#temp2 = np.frombuffer(temp1, dtype=np.int16)
#data = temp2.reshape((wf.getnchannels(),-1), order='F') # If the wav fiel is stereo, then we will get a 2-row numpy array

# Run the html app
if __name__ == '__main__':
    app.run_server(debug=True)


