# Libraries
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
from Formulas_Toolkit import FinancialIndicators
import fmpmodule as fmp
import plotly.graph_objects as go
from dash import html
import dash_bootstrap_components as dbc

'Opcion 1'

# Set up Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1('Financial Data Dashboard'),
    html.Div(className='dropdown-container', style={'display': 'flex', 'flex-direction': 'row'}, children=[
        html.Div(className='dropdown-wrapper', style={'margin-right': '10px'}, children=[
            html.Label('Ticker:'),
            dcc.Dropdown(
                id='ticker-dropdown',
                options=[
                    {'label': 'AAPL', 'value': 'AAPL'},
                    {'label': 'NVDA', 'value': 'NVDA'},
                    {'label': 'GOOGL', 'value': 'GOOGL'},
                    {'label': 'MSFT', 'value': 'MSFT'}
                ],
                value='MSFT',
                style={'width': '200px'}  # ancho de la caja del dropdown
            )
        ]),
        html.Div(className='dropdown-wrapper', style={'margin-right': '10px'}, children=[
            html.Label('Strategy:'),
            dcc.Dropdown(
                id='strategy-dropdown',
                options=[
                    {'label': 'MACD-RSI Signal', 'value': 'macd_rsi_signal'},
                    {'label': 'MACD-MFI Signal', 'value': 'macd_mfi_signal'}
                ],
                value='macd_rsi_signal',
                style={'width': '200px'}  
            )
        ]),
        html.Div(className='dropdown-wrapper', children=[
            html.Label('Date Range:'),
            dcc.DatePickerRange(
                id='date-range',
                display_format='YYYY-MM-DD',
                min_date_allowed=pd.to_datetime('2000-01-01'),
                max_date_allowed=pd.to_datetime('today'),
                start_date=pd.to_datetime('2000-01-01'),
                end_date=pd.to_datetime('today'),
                style={'width': '400px', 'display': 'flex', 'flex-direction': 'row'}  # datepicker
            )
        ])
    ]),
    dcc.Graph(id='graph-output')
])

# Callback to update graph when dropdown or date range is changed
@app.callback(
    Output('graph-output', 'figure'),
    Input('ticker-dropdown', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
    Input('strategy-dropdown', 'value')
)
def update_graph(ticker, start_date, end_date, strategy):
    # Get data for the ticker within the selected date range
    data = fmp.daily_prices(ticker, start_date, end_date)
    data.columns = [col.capitalize() for col in data.columns]
    data = data.loc[:, 'Open':'Volume']
    data = data.sort_index()
    data.index = pd.to_datetime(data.index)

    #  financial indicators 
    analysis = FinancialIndicators(data)
    if strategy == 'macd_rsi_signal':
        signal_data = analysis.macd_rsi_signal(12, 26, 9, 35, 70)
    elif strategy == 'macd_mfi_signal':
        signal_data = analysis.macd_mfi_signal(12, 26, 9, 25, 70)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        showlegend=False,
        name='Price'
    )])

    # buy and sell signals
    signal_data.index = pd.to_datetime(signal_data.index)
    buy_dates = signal_data[signal_data['Signal'] == 1].index
    sell_dates = signal_data[signal_data['Signal'] == -1].index

    fig.add_trace(go.Scatter(
        x=buy_dates,
        y=data.loc[buy_dates]['Close'],
        mode='markers',
        marker=dict(
            size=10,
            color='yellow',
            symbol='triangle-up'
        ),
        showlegend=False,
        name='Buy'
    ))

    fig.add_trace(go.Scatter(
        x=sell_dates,
        y=data.loc[sell_dates]['Close'],
        mode='markers',
        marker=dict(
            size=10,
            color='blue',
            symbol='triangle-down'
        ),
        showlegend=False,
        name='Sell'
    ))

    fig.update_layout(
        title=ticker + ' Stock Price',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        height=600,
        autosize=True,
        hovermode="x"
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=False, port=8055)

    dashboard_html = app.index_string

    with open('dashboard.html', 'w') as f:
        f.write(dashboard_html)

#%%

'Opcion 2'

# Libraries
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
from Formulas_Toolkit import FinancialIndicators
import fmpmodule as fmp
import plotly.graph_objects as go

# Set up Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1('Financial Data Dashboard'),
    html.Div(className='dropdown-container', style={'display': 'flex', 'flex-direction': 'column'}, children=[
        html.Div(style={'margin-bottom': '20px'}, children=[
            html.Div(className='dropdown-wrapper', style={'margin-bottom': '10px'}, children=[
                html.Label('Ticker:'),
                dcc.Dropdown(
                    id='ticker-dropdown',
                    options=[
                        {'label': 'AAPL', 'value': 'AAPL'},
                        {'label': 'NVDA', 'value': 'NVDA'},
                        {'label': 'GOOGL', 'value': 'GOOGL'},
                        {'label': 'MSFT', 'value': 'MSFT'}
                    ],
                    value='MSFT',
                    style={'width': '200px'}  # Adjust the width of the dropdown box
                )
            ]),
            html.Div(className='dropdown-wrapper', style={'margin-bottom': '10px'}, children=[
                html.Label('Strategy:'),
                dcc.Dropdown(
                    id='strategy-dropdown',
                    options=[
                        {'label': 'MACD-RSI Signal', 'value': 'macd_rsi_signal'},
                        {'label': 'MACD-MFI Signal', 'value': 'macd_mfi_signal'}
                    ],
                    value='macd_rsi_signal',
                    style={'width': '200px'}  # Adjust the width of the dropdown box
                )
            ]),
            html.Div(className='dropdown-wrapper', children=[
                html.Label('Date Range:'),
                dcc.DatePickerRange(
                    id='date-range',
                    display_format='YYYY-MM-DD',
                    min_date_allowed=pd.to_datetime('2000-01-01'),
                    max_date_allowed=pd.to_datetime('today'),
                    start_date=pd.to_datetime('2000-01-01'),
                    end_date=pd.to_datetime('today'),
                    style={'width': '400px'}  # Adjust the width of the datepicker
                )
            ])
        ]),
        dcc.Graph(id='graph-output', style={'margin-top': '20px'})
    ])
])

# Callback to update graph when dropdown or date range is changed
@app.callback(
    Output('graph-output', 'figure'),
    Input('ticker-dropdown', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
    Input('strategy-dropdown', 'value')
)
def update_graph(ticker, start_date, end_date, strategy):
    # Get data for the ticker within the selected date range
    data = fmp.daily_prices(ticker, start_date, end_date)
    data.columns = [col.capitalize() for col in data.columns]
    data = data.loc[:, 'Open':'Volume']
    data = data.sort_index()
    data.index = pd.to_datetime(data.index)

    # Apply financial indicators based on the selected strategy
    analysis = FinancialIndicators(data)
    if strategy == 'macd_rsi_signal':
        signal_data = analysis.macd_rsi_signal(12, 26, 9, 35, 70)
    elif strategy == 'macd_mfi_signal':
        signal_data = analysis.macd_mfi_signal(12, 26, 9, 25, 70)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Create the candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        showlegend=False,
        name='Price'
    )])

    # Mark buy and sell signals on the chart
    signal_data.index = pd.to_datetime(signal_data.index)
    buy_dates = signal_data[signal_data['Signal'] == 1].index
    sell_dates = signal_data[signal_data['Signal'] == -1].index

    fig.add_trace(go.Scatter(
        x=buy_dates,
        y=data.loc[buy_dates]['Close'],
        mode='markers',
        marker=dict(
            size=10,
            color='yellow',
            symbol='triangle-up'
        ),
        showlegend=False,
        name='Buy'
    ))

    fig.add_trace(go.Scatter(
        x=sell_dates,
        y=data.loc[sell_dates]['Close'],
        mode='markers',
        marker=dict(
            size=10,
            color='blue',
            symbol='triangle-down'
        ),
        showlegend=False,
        name='Sell'
    ))

    fig.update_layout(
        title=ticker + ' Stock Price',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_dark',
        height=600,
        autosize=True,
        hovermode="x"
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=False, port=8053)

    dashboard_html = app.index_string

    with open('dashboard.html', 'w') as f:
        f.write(dashboard_html)

