# coding: utf-8
"""
@author: michelebradley
"""

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import os
import numpy as np
import base64
from app import app
from py_scripts import multivariate, descriptive_statistics, LSTM, mongoQueryScripts, fb_prophet, supply_demand_data

# get prices
ng_df = mongoQueryScripts.ng_df
ng_df = ng_df.sort_values(by=['month_timestamp'])
ng_df_stats = pd.DataFrame({"label": ["Minimum", "Maximum", "Mean", "Standard Deviation", "Variance"],
                           "value": [float(ng_df["ng_val"].min()), float(ng_df["ng_val"].max()), int(ng_df["ng_val"].mean()), int(ng_df["ng_val"].std()), int(ng_df["ng_val"].var())]})
ng_df['month_timestamp'] =  pd.to_datetime(ng_df['month_timestamp'], infer_datetime_format=True)

#multivariate data
ols_df, f_df, fa_df, residual_ols, plot_position_ols, trade_ols = multivariate.getOLS(sym = "ng")

# get rig_information
ng_data, ticker = supply_demand_data.get_supply_demand_data('ng')
df_stats =  descriptive_statistics.descriptive_stats(ng_data)
df_stats_df = df_stats.stats_df()
ng_stats = df_stats_df.reset_index().round(2)

# LSTM run
LSTM_prediction, LSTM_future, LSTM_all_data, LSTM_residuals, recommendations = LSTM.LSTM_prediction("ng")
LSTM_prediction_week = LSTM_future.tail(7).round(2)

# FB Prophet Data
ng_daily_df, fb_yu, low, medium, high = fb_prophet.testRunProphet('ng')
ng_daily_df = ng_daily_df.sort_values(by=['day_timestamp'])
headers = ["ds", 'yhat']
fb_prophet_forecast = medium.reset_index()
fb_prophet_prediction = fb_prophet_forecast[headers].tail(7).round(2)
ng_daily_df_stats = pd.DataFrame({"label": ["Minimum", "Maximum", "Mean", "Standard Deviation", "Variance"],
                           "value": [float(ng_daily_df["ng_val"].min()), float(ng_daily_df["ng_val"].max()), int(ng_daily_df["ng_val"].mean()), int(ng_daily_df["ng_val"].std()), int(ng_daily_df["ng_val"].var())]})

# reusable componenets
def make_dash_table(df):
    ''' Return a dash definitio of an HTML table for a Pandas dataframe '''
    table = []
    for index, row in df.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table

def print_button():
    printButton = html.A(['Print PDF'],className="button no-print print",style={'position': "absolute", 'top': '-40', 'right': '0'})
    return printButton

# includes page/full view
def get_logo():

    logo = html.Div([

        html.Div([
            html.Img(src='https://raw.githubusercontent.com/sjv1030/data602-finalproject/master/logo.png?token=AbaIfbMXgUWBjbUl0XFIbJ5GhI9S_MHSks5bCjRRwA%3D%3D', height = 80, width = 120)
        ], className="ten columns padded"),

        html.Div([
            dcc.Link('Click Here for Oil Analysis   ', href='/apps/app_oil/overview')
        ], className="two columns page-view no-print")


    ], className="row gs-header")
    return logo

def get_header():
    header = html.Div([

        html.Div([
            html.H5(
                'Analysis of Natural Gas Futures')
        ], className="twelve columns padded")
    ], className="row gs-header gs-text-header")
    return header


def get_menu():
    menu = html.Div([
        dcc.Link('Overview   ', href='/apps/app_ng/overview', className="tab first"),
        dcc.Link('Multivariable Model   ', href='/apps/app_ng/multivariable', className="tab"),
        dcc.Link('FB Prophet   ', href='/apps/app_ng/fb_prophet', className="tab"),
        dcc.Link('Long Short-Term Memory   ', href='/apps/app_ng/LSTM', className="tab"),
        dcc.Link('Takeaways   ', href='/apps/app_ng/takeaways', className="tab")
    ], className="row ")
    return menu

## Page layouts
overview = html.Div([  # page 1

        print_button(),

        html.Div([

            # Header
            get_logo(),
            get_header(),
            html.Br([]),
            get_menu(),
            # Row 3
            html.Div([
                html.Div([
                    html.H6('Natural Gas Summary',
                            className="gs-header gs-text-header padded"),
                    html.Br([]),
                    html.P("\
                            The futures market is characterized by the ability to use very \
                            high leverage relative to stock markets. Futures fundamental \
                            analyis requires heavy research into the underlying factors that \
                            determine the price level for a financial asset or commodity.\
                            Therefore most traders tend to analyze only one or two futures at a \
                            time. This analysis utilizes machine learning techniques to better \
                            comprehend the change in prices for Natural Gas Markets. We will be \
                            utilzing various external values related to this market as well as \
                            global economies, along with various prediction techniques."),

                ], className="six columns"),

                html.Div([
                    html.H6(["Natural Gas Spot Data Statistics"],
                            className="gs-header gs-table-header padded"),
                    html.Table(make_dash_table(ng_df_stats))
                ], className="six columns"),

            ], className="row "),

            # Row 4

            html.Div([
                html.Div([
                    html.H6('Natural Gas Spot Prices Over the Years',
                            className="gs-header gs-text-header padded"),
                    dcc.Graph(
                        id = "graph-1",
                        figure={
                            'data': [
                                go.Scatter(
                                    x = ng_df['month_timestamp'],
                                    y = ng_df['ng_val'],
                                    line = {"color": "rgb(53, 83, 255)"},
                                    mode = "lines")
                            ],
                            'layout': go.Layout(autosize = False,
                            title = "",
                            font = {"family": "Raleway","size": 10},
                            height = 200,
                            width = 340,
                            hovermode = "closest",
                            margin = {
                              "r": 20,
                              "t": 20,
                              "b": 20,
                              "l": 50
                            },
                        )
                        },
                        config={
                            'displayModeBar': False
                        }
                    )
                ], className="six columns"),

                html.Div([
                    html.H6("US Economic Data",
                            className="gs-header gs-table-header padded"),
                    dcc.Graph(
                        id="graph-2",
                        figure={
                            'data': [
                                go.Scatter(
                                    x=ng_data.index, # assign x as the dataframe column 'x'
                                    y=ng_data['twd'],
                                    name = 'Daily Trade-weighted Dollar Index'
                                ),
                                go.Scatter(
                                    x=ng_data.index, # assign x as the dataframe column 'x'
                                    y=ng_data['ip'],
                                    name = 'Monthly US Industrial Production'
                                )
                            ],
                            'layout': go.Layout(
                                autosize = False,
                                title = "",
                                font = {"family": "Raleway","size": 10},
                                height = 200,
                                width = 340,
                                hovermode = "closest",
                                legend = {"x": -0.0277108433735,"y": -0.142606516291,"orientation": "h"},
                                margin = {
                                  "r": 20,
                                  "t": 20,
                                  "b": 20,
                                  "l": 50
                                },
                                showlegend = True,
                            )
                        },
                        config={
                            'displayModeBar': False
                        }
                    )
                ], className="six columns"),
            ], className="row "),

            # Row 5
            html.Div([
                html.Div([
                    html.H6('Natural Gas Production Statistics',
                            className="gs-header gs-table-header padded"),
                    html.Table(make_dash_table(ng_stats))
                ], className="six columns"),
                html.Div([
                    html.H6("Natural Gas Information",
                            className="gs-header gs-table-header padded"),
                    dcc.Graph(
                        id='graph-3',
                        figure={
                            'data': [
                                go.Scatter(
                                    x=ng_data.index, # assign x as the dataframe column 'x'
                                    y=ng_data['rig'],
                                    name = 'Rigs'
                                ),
                                go.Scatter(
                                    x=ng_data.index, # assign x as the dataframe column 'x'
                                    y=ng_data['prod'],
                                    name = 'Production'
                                ),
                                go.Scatter(
                                    x=ng_data.index, # assign x as the dataframe column 'x'
                                    y=ng_data['cons'],
                                    name = 'Consumption'
                                )
                            ],
                            'layout': go.Layout(
                                autosize = False,
                                title = "",
                                font = {"family": "Raleway","size": 10},
                                height = 200,
                                width = 340,
                                hovermode = "closest",
                                legend = {"x": -0.0277108433735,"y": -0.142606516291,"orientation": "h"},
                                margin = {
                                  "r": 20,
                                  "t": 20,
                                  "b": 20,
                                  "l": 50
                                },
                                showlegend = True,
                            )
                        },
                        config={
                            'displayModeBar': False
                        }
                    )
                ], className="six columns"),

            ], className="row ")

        ], className="subpage")

    ], className="page")


multivariable = html.Div([  # page 2

        print_button(),

        html.Div([

            # Header
            get_logo(),
            get_header(),
            html.Br([]),
            get_menu(),
            # Row 3
            html.Div([
                html.Div([
                    html.H6('Multivariate Model Summary',
                            className="gs-header gs-text-header padded"),
                    html.Br([]),
                    html.P("\
                            Monthly WTI oil prices were fitted on fundamental data (eg, rig count,\
                            inventory, production), economic data (eg, trade-weighted data, industrial\
                            production), and seasonal variables along with interaction\
                            terms to forecast future prices. This is a multivariable ARIMA model\
                            that utilizes many variables. "),

                ], className="six columns"),

                html.Div([
                    html.H6(["Natural Gas Prediction Values"],
                            className="gs-header gs-table-header padded"),
                    html.Table(make_dash_table(f_df.reset_index().round(2)))
                ], className="six columns"),

            ], className="row "),

            # Row 4

            html.Div([
                html.Div([
                    html.H6('Natural Gas Spot Prediction',
                            className="gs-header gs-text-header padded"),
                    dcc.Graph(
                        id = "graph-1",
                        figure={
                            'data': [
                                go.Scatter(
                                    x = ng_df['month_timestamp'],
                                    y = ng_df['ng_val'],
                                    line = {"color": "rgb(53, 83, 255)"},
                                    name = "Actual",
                                    mode = "lines"),
                                go.Scatter(
                                    x = f_df['month_timestamp'],
                                    y = f_df['ng_val'],
                                    line = {"color": "rgb(255, 0, 0)"},
                                    name = "Predicted",
                                    mode = "lines"),
                            ],
                            'layout': go.Layout(autosize = False,
                            title = "",
                            font = {"family": "Raleway","size": 10},
                            height = 200,
                            width = 340,
                            hovermode = "closest",
                            legend = {"x": -0.0277108433735,"y": -0.142606516291,"orientation": "h"},
                            margin = {
                              "r": 20,
                              "t": 20,
                              "b": 20,
                              "l": 50
                            },
                        )
                        },
                        config={
                            'displayModeBar': False
                        }
                    )
                ], className="six columns"),

                html.Div([
                    html.H6("Residual Plot",
                            className="gs-header gs-table-header padded"),
                    dcc.Graph(
                        id="graph-2",
                        figure={
                            'data': [
                                go.Scatter(
                                    x = np.arange(-2, 4, len(range(-2, 4))/len(residual_ols)),
                                    y = residual_ols,
                                    line = {"color": "rgb(53, 83, 255)"},
                                    mode = "markers")
                            ],
                            'layout': go.Layout(autosize = False,
                            title = "",
                            font = {"family": "Raleway","size": 10},
                            height = 200,
                            width = 340,
                            hovermode = "closest",
                            margin = {
                              "r": 20,
                              "t": 20,
                              "b": 20,
                              "l": 50
                            },
                        )
                        },
                        config={
                            'displayModeBar': False
                        }
                    )
                ], className="six columns"),
            ], className="row "),

            # Row 5
            html.Div([
                html.Div([
                    html.H6('Model Statistics (predictor, beta, p-value)',
                            className="gs-header gs-table-header padded"),
                    html.Table(make_dash_table(ols_df.reset_index().round(2)))
                ], className="six columns"),
                html.Div([
                    html.H6("Q-Q Plot",
                            className="gs-header gs-table-header padded"),
                    dcc.Graph(
                        id='graph-3',
                        figure={
                            'data': [
                                go.Scatter(
                                    x = np.arange(-2, 4, len(range(-2, 4))/len(residual_ols)),
                                    y = plot_position_ols[1],
                                    line = {"color": "rgb(53, 83, 255)"},
                                    mode = "markers")
                            ],
                            'layout': go.Layout(autosize = False,
                            title = "",
                            font = {"family": "Raleway","size": 10},
                            yaxis = {"rangemode": "tozero",
                              "showline": True,
                              "nticks": 11,
                              "showgrid": True,
                              "title": "Y Axis"},
                            xaxis = {
                              "rangemode": "tozero",
                              "showline": True,
                              "showgrid": True,
                              "title": "X Axis"
                            },
                            height = 200,
                            width = 340,
                            hovermode = "closest",
                            margin = {
                              "r": 20,
                              "t": 20,
                              "b": 20,
                              "l": 50
                            },
                        )
                        },
                        config={
                            'displayModeBar': False
                        }
                    )
                ], className="six columns"),

            ], className="row ")

        ], className="subpage")

    ], className="page")


fbProphet = html.Div([ # page 3

        print_button(),

        html.Div([

            # Header
            get_logo(),
            get_header(),
            html.Br([]),
            get_menu(),
            # Row 3
            html.Div([
                html.Div([
                    html.H6('FB Prophet Summary',
                            className="gs-header gs-text-header padded"),
                    html.Br([]),
                    html.P("\
                            FBProphet is a recently open-sourced ML tool optimized for time series\
                             fitting and prediction. It intrinsically discovers and optimizes for \
                             multiple seasonality and changing growth rates, called changepoints. \
                             In this deployment, the ability to add other time series regressors is\
                              exploited; regressors are added to the linear part of the model. The \
                              daily values of the commodity are regressed on the rolling 9-day \
                              close-to-close volatility - a volatility measure appropriate for our \
                              single daily scalar data points. Historical volatility was studied in \
                              30-day windows: values were binned in incremental volatility ranges, \
                              and three scenario paths were defined by measuring the frequencies. We \
                              simulated three volatility series with the chracterstic bin frequencies, \
                              then predicted commodity values for those scenarios."),

                ], className="six columns"),

                html.Div([
                    html.H6(["Natural Gas Spot Data Statistics"],
                            className="gs-header gs-table-header padded"),
                    html.Table(make_dash_table(ng_daily_df_stats))
                ], className="six columns"),

            ], className="row "),

            # Row 4

            html.Div([
                html.Div([
                    html.H6('Natural Gas Spot Predicted Using FB Prophet',
                            className="gs-header gs-text-header padded"),
                    dcc.Graph(
                        id = "graph-1",
                        figure={
                            'data': [
                                go.Scatter(
                                    x=ng_daily_df.tail(529)['day_timestamp'],
                                    y=ng_daily_df.tail(529)['ng_val'],
                                    mode = "lines",
                                    name = 'Actual'),
                                go.Scatter(
                                    x = low['ds'][-30:],
                                    y = low['yhat'][-30:],
                                    mode = "lines",
                                    name = 'low'),
                                go.Scatter(
                                    x = medium['ds'][-30:],
                                    y = medium['yhat'][-30:],
                                    mode = "lines",
                                    name = 'med'),
                                go.Scatter(
                                    x = high['ds'][-30:],
                                    y = high['yhat'][-30:],
                                    mode = "lines",
                                    name = 'high')
                            ],
                            'layout': go.Layout(autosize = False,
                            title = "",
                            font = {"family": "Raleway","size": 10},
                            height = 200,
                            width = 340,
                            hovermode = "closest",
                            legend = {"x": -0.0277108433735,"y": -0.142606516291,"orientation": "h"},
                            margin = {
                              "r": 20,
                              "t": 20,
                              "b": 20,
                              "l": 50
                            },
                        )
                        },
                        config={
                            'displayModeBar': False
                        }
                    )
                ], className="six columns"),

                html.Div([
                    html.H6("Rolling 9-day Volatility",
                            className="gs-header gs-table-header padded"),
                    dcc.Graph(
                        id="graph-2",
                        figure={
                            'data': [
                                go.Scatter(
                                    x = ng_daily_df['day_timestamp'],
                                    y = fb_yu[-400:],
                                    line = {"color": "rgb(53, 83, 255)"},
                                    mode = "lines")
                            ],
                            'layout': go.Layout(autosize = False,
                            title = "",
                            font = {"family": "Raleway","size": 10},
                            height = 200,
                            width = 340,
                            hovermode = "closest",
                            margin = {
                              "r": 20,
                              "t": 20,
                              "b": 20,
                              "l": 50
                            },
                        )
                        },
                        config={
                            'displayModeBar': False
                        }
                    )
                ], className="six columns"),
            ], className="row "),

            # Row 5
            html.Div([
                html.Div([
                    html.H6('Natural Gas Prediction Prices',
                            className="gs-header gs-table-header padded"),
                    html.Table(make_dash_table(fb_prophet_prediction))
                ], className="six columns"),
                html.Div([
                    html.H6("Actual vs Prediction",
                            className="gs-header gs-table-header padded"),
                    dcc.Graph(
                        id='graph-3',
                        figure={
                            'data': [
                                go.Scatter(
                                    x=ng_daily_df.tail(500)['day_timestamp'],
                                    y=ng_daily_df.tail(500)['ng_val'],
                                    name = 'Actual',
                                    mode = "lines"
                                ),
                                go.Scatter(
                                    x= low.head(500)['ds'],
                                    y=low.head(500)['yhat'],
                                    name = 'Prediction',
                                    mode = "lines"
                                )
                            ],
                            'layout': go.Layout(
                                autosize = False,
                                title = "",
                                font = {"family": "Raleway","size": 10},
                                height = 200,
                                width = 340,
                                hovermode = "closest",
                                legend = {"x": -0.0277108433735,"y": -0.142606516291,"orientation": "h"},
                                margin = {
                                  "r": 20,
                                  "t": 20,
                                  "b": 20,
                                  "l": 50
                                },
                                showlegend = True,
                            )
                        },
                        config={
                            'displayModeBar': False
                        }
                    )
                ], className="six columns"),

            ], className="row ")

        ], className="subpage")

        ], className="page")

LSTM = html.Div([  # page 4
        print_button(),

        html.Div([

            # Header
            get_logo(),
            get_header(),
            html.Br([]),
            get_menu(),
            # Row 3
            html.Div([
                html.Div([
                    html.H6('Long Short Term Memory Summary',
                            className="gs-header gs-text-header padded"),
                    html.Br([]),
                    html.P("\
                            LSTM models were first created to help preserve the error that can \
                            be backpropagated through time and layers. This addresses the \
                            vanishing or exploding derivative problem present with RNN models \
                            and they allow recurrent nets to learn many time steps over. Therefore,\
                            LSTMs can learn outside the normal flow and “determine whether to let \
                            new input in or erase the present data” or even whether to perform\
                            addition or multiplication during the data transformation (DL4J). \
                            Erasing data helps the neural network not not overfit our results,\
                            bringing in new data and not forcing relationships between different\
                             datasets that may not make much sense"),

                ], className="six columns"),

                html.Div([
                    html.H6(["Natural Gas Spot Data Statistics"],
                            className="gs-header gs-table-header padded"),
                    html.Table(make_dash_table(ng_df_stats))
                ], className="six columns"),

            ], className="row "),

            # Row 4

            html.Div([
                html.Div([
                    html.H6('Natural Gas Spot Predicted Using LSTM',
                            className="gs-header gs-text-header padded"),
                    dcc.Graph(
                        id = "graph-1",
                        figure={
                            'data': [
                                go.Scatter(
                                    x = LSTM_all_data['month_timestamp'],
                                    y = LSTM_all_data['ng_val'],
                                    line = {"color": "rgb(53, 83, 255)"},
                                    mode = "lines",
                                    name = 'Actual'),
                                go.Scatter(
                                    x = LSTM_all_data['month_timestamp'][-7:],
                                    y = LSTM_all_data['ng_val'][-7:],
                                    line = {"color": "rgb(255, 0, 0)"},
                                    mode = "lines",
                                    name = 'Predicted')
                            ],
                            'layout': go.Layout(autosize = False,
                            title = "",
                            font = {"family": "Raleway","size": 10},
                            height = 200,
                            width = 340,
                            hovermode = "closest",
                            legend = {"x": -0.0277108433735,"y": -0.142606516291,"orientation": "h"},
                            margin = {
                              "r": 20,
                              "t": 20,
                              "b": 20,
                              "l": 50
                            },
                        )
                        },
                        config={
                            'displayModeBar': False
                        }
                    )
                ], className="six columns"),

                html.Div([
                    html.H6("Residual Plot",
                            className="gs-header gs-table-header padded"),
                    dcc.Graph(
                        id="graph-2",
                        figure={
                            'data': [
                                go.Scatter(
                                    x = np.arange(-2, 4, len(range(-2, 4))/len(LSTM_residuals)),
                                    y = LSTM_residuals,
                                    line = {"color": "rgb(53, 83, 255)"},
                                    mode = "markers")
                            ],
                            'layout': go.Layout(autosize = False,
                            title = "",
                            font = {"family": "Raleway","size": 10},
                            height = 200,
                            width = 340,
                            hovermode = "closest",
                            margin = {
                              "r": 20,
                              "t": 20,
                              "b": 20,
                              "l": 50
                            },
                        )
                        },
                        config={
                            'displayModeBar': False
                        }
                    )
                ], className="six columns"),
            ], className="row "),

            # Row 5
            html.Div([
                html.Div([
                    html.H6('Oil Prediction Prices',
                            className="gs-header gs-table-header padded"),
                    html.Table(make_dash_table(LSTM_prediction_week))
                ], className="six columns"),
                html.Div([
                    html.H6("Actual vs Prediction",
                            className="gs-header gs-table-header padded"),
                    dcc.Graph(
                        id='graph-3',
                        figure={
                            'data': [
                                go.Scatter(
                                    x=ng_df['month_timestamp'][-7:],
                                    y=ng_df['ng_val'][-7:],
                                    name = 'Actual',
                                    mode = "markers"
                                ),
                                go.Scatter(
                                    x=LSTM_prediction['month_timestamp'][-7:],
                                    y=LSTM_prediction['ng_val'],
                                    name = 'Prediction',
                                    mode = "markers"
                                )
                            ],
                            'layout': go.Layout(
                                autosize = False,
                                title = "",
                                font = {"family": "Raleway","size": 10},
                                height = 200,
                                width = 340,
                                hovermode = "closest",
                                legend = {"x": -0.0277108433735,"y": -0.142606516291,"orientation": "h"},
                                margin = {
                                  "r": 20,
                                  "t": 20,
                                  "b": 20,
                                  "l": 50
                                },
                                showlegend = True,
                            )
                        },
                        config={
                            'displayModeBar': False
                        }
                    )
                ], className="six columns"),

            ], className="row ")

        ], className="subpage")

        ], className="page")


takeaways = html.Div([  # page 5

        print_button(),

        html.Div([

            # Header

            get_logo(),
            get_header(),
            html.Br([]),
            get_menu(),

            # Row 1

            html.Div([

                html.Div([
                    html.H6('Recommendations',
                            className="gs-header gs-text-header padded"),
                    html.Br([]),
                    html.P('Multivariate Model recommends that you:'),
                    html.Br([]),
                    html.P('BUY Natural Gas Futures'),
                    html.Br([]),
                    html.P('FB Prophet Model recommends that you:'),
                    html.Br([]),
                    html.P('BUY Natural Gas Futures'),
                    html.Br([]),
                    html.P('Long Short Term Memory Model recommends that you:'),
                    html.Br([]),
                    html.P('BUY Natural Gas Futures'),
                    html.Br([])
                ], className="six columns"),

                html.Div([
                    html.H6("About Future Analytics Recommendations",
                            className="gs-header gs-table-header padded"),
                    html.Br([]),
                    html.P('We base our recommendations and prediction models on many factors that are \
                    well researched within the data science and data analytics fields*'),
                    html.Br([]),
                    html.P("Recommendations are given based on not just one value in our prediction, \
                    but the overall trend we are seeing with our prediction models."),
                    html.Br([]),
                    html.P("Note that the Multivariate and FB Prophet models both utilize variables\
                     related to the climate of the US Economy or of the Natural Gas Industry as a whole.\
                     Both include a lag time on the other variables to generate a better estimate on \
                     the price of the future in the months to come."),
                     html.Br([]),
                     html.P("The Long Short Term Memory model however, does not \
                     utilize any other variables other than the trend and price of the dataset.\
                     This is a machine learning techique that attempts to find trends in the data \
                     to determine future prices."),
                      html.Br([]),
                      html.P("* Note that we do not expect any of these predictions to be \
                      right on the money, but we do believe these statistical models can \
                      provide insight on the trends for Natural Gas pricing in the months to come.")
                ], className="six columns"),

            ], className="row ")

        ], className="subpage")

    ], className="page")

noPage = html.Div([  # 404

    html.P(["404 Page not found"])

    ], className="no-page")



# Describe the layout, or the UI, of the app
layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

external_css = ["https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
                "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
                "//fonts.googleapis.com/css?family=Raleway:400,300,600",
                "https://codepen.io/bcd/pen/KQrXdb.css",
                "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"]

for css in external_css:
    app.css.append_css({"external_url": css})

external_js = ["https://code.jquery.com/jquery-3.2.1.min.js",
               "https://codepen.io/bcd/pen/YaXojL.js"]

for js in external_js:
    app.scripts.append_script({"external_url": js})


if __name__ == '__main__':
    app.run_server(debug=True)
