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

from app import app
from py_scripts import multivariate_michele, descriptive_statistics

#app = dash.Dash(__name__)
#server = app.server

#app.config['suppress_callback_exceptions']=True

# Holders for Other dataframes
df_equity_char = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 3, 4, 5]})
df_equity_diver =  pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 3, 4, 5]})
df_expenses =  pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 3, 4, 5]})
df_minimums =  pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 3, 4, 5]})
df_dividend =  pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 3, 4, 5]})
df_realized =  pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 3, 4, 5]})
df_unrealized =  pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 3, 4, 5]})

df_graph = pd.DataFrame({"Date": [1, 2, 3, 4], 'Vanguard 500 Index Fund': [2, 3, 4, 5], 'MSCI EAFE Index Fund (ETF)':[4, 5, 6, 7]})

# get prices
wti_d = pd.read_csv("py_data/ng_values.csv")
wti_d_stats = pd.DataFrame({"label": ["Minimum", "Maximum", "Mean", "Standard Deviation", "Variance"],
                           "value": [float(wti_d["nat_gas"].min()), float(wti_d["nat_gas"].max()), int(wti_d["nat_gas"].mean()), int(wti_d["nat_gas"].std()), int(wti_d["nat_gas"].var())]})

#multivatiate-ARIMA data
data_silverio, residual_arima, plot_position_arima = multivariate_michele.ng_multivariable()

# get us economic data
us_econ = pd.read_csv("py_data/us_econ.csv")

# get rig_information
ng_data = pd.read_csv("py_data/ng_data.csv")
df_stats =  descriptive_statistics.descriptive_stats(ng_data)
df_stats_df = df_stats.stats_df()
ng_stats = df_stats_df.reset_index().round(2)

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
            html.Img(src='http://logonoid.com/images/vanguard-logo.png', height='40', width='160')
        ], className="ten columns padded"),

        html.Div([
            dcc.Link('Full View   ', href='/full-view')
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
        dcc.Link('ARIMA   ', href='/apps/app_ng/multivariable_ARIMA', className="tab"),
        dcc.Link('FB Prophet   ', href='/apps/app_ng/fb_prophet', className="tab"),
        dcc.Link('Recurrent Neural Networks', href='/apps/app_ng/RNN', className="tab"),
        dcc.Link('Support Vector Machine  ', href='/apps/app_ng/SVM', className="tab"),
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
                    html.Table(make_dash_table(wti_d_stats))
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
                                    x = wti_d.index,
                                    y = wti_d['nat_gas'],
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
                                    x=us_econ.index, # assign x as the dataframe column 'x'
                                    y=us_econ['twd'],
                                    name = 'Daily Trade-weighted Dollar Index'
                                ),
                                go.Scatter(
                                    x=us_econ.index, # assign x as the dataframe column 'x'
                                    y=us_econ['ip'],
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


multivariable_ARIMA = html.Div([  # page 2

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
                    html.H6('Multivariate ARIMA',
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
                    html.Table(make_dash_table(wti_d_stats))
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
                                    x = wti_d.index,
                                    y = wti_d['nat_gas'],
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
                    html.H6("Residual Plot",
                            className="gs-header gs-table-header padded"),
                    dcc.Graph(
                        id="graph-2",
                        figure={
                            'data': [
                                go.Scatter(
                                    x = np.arange(-2, 4, len(range(-2, 4))/len(residual_arima)),
                                    y = residual_arima,
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
                    html.H6('Multivariate ARIMA Statistics',
                            className="gs-header gs-table-header padded"),
                    html.Table(make_dash_table(ng_stats))
                ], className="six columns"),
                html.Div([
                    html.H6("Q-Q Plot",
                            className="gs-header gs-table-header padded"),
                    dcc.Graph(
                        id='graph-3',
                        figure={
                            'data': [
                                go.Scatter(
                                    x = np.arange(-2, 4, len(range(-2, 4))/len(residual_arima)),
                                    y = plot_position_arima[1],
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

            # Row 1

            html.Div([

                html.Div([
                    html.H6(["Portfolio"],
                            className="gs-header gs-table-header padded")
                ], className="twelve columns"),

            ], className="row "),

            # Row 2

            html.Div([

                html.Div([
                    html.Strong(["Stock style"]),
                    dcc.Graph(
                        id='graph-5',
                        figure={
                            'data': [
                                go.Scatter(
                                    x = ["1"],
                                    y = ["1"],
                                    hoverinfo = "none",
                                    marker = {
                                        "color": ["transparent"]
                                    },
                                    mode = "markers",
                                    name = "B",
                                )
                            ],
                            'layout': go.Layout(
                                title = "",
                                annotations = [
                                {
                                  "x": 0.990130093458,
                                  "y": 1.00181709504,
                                  "align": "left",
                                  "font": {
                                    "family": "Raleway",
                                    "size": 9
                                  },
                                  "showarrow": False,
                                  "text": "<b>Market<br>Cap</b>",
                                  "xref": "x",
                                  "yref": "y"
                                },
                                {
                                  "x": 1.00001816013,
                                  "y": 1.35907755794e-16,
                                  "font": {
                                    "family": "Raleway",
                                    "size": 9
                                  },
                                  "showarrow": False,
                                  "text": "<b>Style</b>",
                                  "xref": "x",
                                  "yanchor": "top",
                                  "yref": "y"
                                }
                              ],
                              autosize = False,
                              width = 200,
                              height = 150,
                              hovermode = "closest",
                              margin = {
                                "r": 30,
                                "t": 20,
                                "b": 20,
                                "l": 30
                              },
                              shapes = [
                                {
                                  "fillcolor": "rgb(127, 127, 127)",
                                  "line": {
                                    "color": "rgb(0, 0, 0)",
                                    "width": 2
                                  },
                                  "opacity": 0.3,
                                  "type": "rectangle",
                                  "x0": 0,
                                  "x1": 0.33,
                                  "xref": "paper",
                                  "y0": 0,
                                  "y1": 0.33,
                                  "yref": "paper"
                                },
                                {
                                  "fillcolor": "rgb(127, 127, 127)",
                                  "line": {
                                    "color": "rgb(0, 0, 0)",
                                    "dash": "solid",
                                    "width": 2
                                  },
                                  "opacity": 0.3,
                                  "type": "rectangle",
                                  "x0": 0.33,
                                  "x1": 0.66,
                                  "xref": "paper",
                                  "y0": 0,
                                  "y1": 0.33,
                                  "yref": "paper"
                                },
                                {
                                  "fillcolor": "rgb(127, 127, 127)",
                                  "line": {
                                    "color": "rgb(0, 0, 0)",
                                    "width": 2
                                  },
                                  "opacity": 0.3,
                                  "type": "rectangle",
                                  "x0": 0.66,
                                  "x1": 0.99,
                                  "xref": "paper",
                                  "y0": 0,
                                  "y1": 0.33,
                                  "yref": "paper"
                                },
                                {
                                  "fillcolor": "rgb(127, 127, 127)",
                                  "line": {
                                    "color": "rgb(0, 0, 0)",
                                    "width": 2
                                  },
                                  "opacity": 0.3,
                                  "type": "rectangle",
                                  "x0": 0,
                                  "x1": 0.33,
                                  "xref": "paper",
                                  "y0": 0.33,
                                  "y1": 0.66,
                                  "yref": "paper"
                                },
                                {
                                  "fillcolor": "rgb(127, 127, 127)",
                                  "line": {
                                    "color": "rgb(0, 0, 0)",
                                    "width": 2
                                  },
                                  "opacity": 0.3,
                                  "type": "rectangle",
                                  "x0": 0.33,
                                  "x1": 0.66,
                                  "xref": "paper",
                                  "y0": 0.33,
                                  "y1": 0.66,
                                  "yref": "paper"
                                },
                                {
                                  "fillcolor": "rgb(127, 127, 127)",
                                  "line": {
                                    "color": "rgb(0, 0, 0)",
                                    "width": 2
                                  },
                                  "opacity": 0.3,
                                  "type": "rectangle",
                                  "x0": 0.66,
                                  "x1": 0.99,
                                  "xref": "paper",
                                  "y0": 0.33,
                                  "y1": 0.66,
                                  "yref": "paper"
                                },
                                {
                                  "fillcolor": "rgb(127, 127, 127)",
                                  "line": {
                                    "color": "rgb(0, 0, 0)",
                                    "width": 2
                                  },
                                  "opacity": 0.3,
                                  "type": "rectangle",
                                  "x0": 0,
                                  "x1": 0.33,
                                  "xref": "paper",
                                  "y0": 0.66,
                                  "y1": 0.99,
                                  "yref": "paper"
                                },
                                {
                                  "fillcolor": "rgb(255, 127, 14)",
                                  "line": {
                                    "color": "rgb(0, 0, 0)",
                                    "width": 1
                                  },
                                  "opacity": 0.9,
                                  "type": "rectangle",
                                  "x0": 0.33,
                                  "x1": 0.66,
                                  "xref": "paper",
                                  "y0": 0.66,
                                  "y1": 0.99,
                                  "yref": "paper"
                                },
                                {
                                  "fillcolor": "rgb(127, 127, 127)",
                                  "line": {
                                    "color": "rgb(0, 0, 0)",
                                    "width": 2
                                  },
                                  "opacity": 0.3,
                                  "type": "rectangle",
                                  "x0": 0.66,
                                  "x1": 0.99,
                                  "xref": "paper",
                                  "y0": 0.66,
                                  "y1": 0.99,
                                  "yref": "paper"
                                }
                              ],
                              xaxis = {
                                "autorange": True,
                                "range": [0.989694747864, 1.00064057995],
                                "showgrid": False,
                                "showline": False,
                                "showticklabels": False,
                                "title": "<br>",
                                "type": "linear",
                                "zeroline": False
                              },
                              yaxis = {
                                "autorange": True,
                                "range": [-0.0358637178721, 1.06395696354],
                                "showgrid": False,
                                "showline": False,
                                "showticklabels": False,
                                "title": "<br>",
                                "type": "linear",
                                "zeroline": False
                              }
                            )
                        },
                        config={
                            'displayModeBar': False
                        }
                    )

                ], className="four columns"),

                html.Div([
                    html.P("Vanguard 500 Index Fund seeks to track the performance of\
                     a benchmark index that meaures the investment return of large-capitalization stocks."),
                    html.P("Learn more about this portfolio's investment strategy and policy.")
                ], className="eight columns middle-aligned"),

            ], className="row "),

            # Row 3

            html.Br([]),

            html.Div([

                html.Div([
                    html.H6(["Equity characteristics as of 01/31/2018"], className="gs-header gs-table-header tiny-header"),
                    html.Table(make_dash_table(df_equity_char), className="tiny-header")
                ], className=" twelve columns"),

            ], className="row "),

            # Row 4

            html.Div([

                html.Div([
                    html.H6(["Equity sector diversification"], className="gs-header gs-table-header tiny-header"),
                    html.Table(make_dash_table(df_equity_diver), className="tiny-header")
                ], className=" twelve columns"),

            ], className="row "),

        ], className="subpage")

    ], className="page")

RNN = html.Div([  # page 4

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
                    html.H6(["Expenses"],
                            className="gs-header gs-table-header padded")
                ], className="twelve columns"),

            ], className="row "),

            # Row 2

            html.Div([

                html.Div([
                    html.Strong(),
                    html.Table(make_dash_table(df_expenses)),
                    html.H6(["Minimums"],
                            className="gs-header gs-table-header padded"),
                    html.Table(make_dash_table(df_minimums))
                ], className="six columns"),

                html.Div([
                    html.Br([]),
                    html.Strong("Fees on $10,000 invested over 10 years"),
                    dcc.Graph(
                        id = 'graph-6',
                        figure = {
                            'data': [
                                go.Bar(
                                    x = ["Category Average", "This fund"],
                                    y = ["2242", "329"],
                                    marker = {"color": "rgb(53, 83, 255)"},
                                    name = "A"
                                ),
                                go.Bar(
                                    x = ["This fund"],
                                    y = ["1913"],
                                    marker = {"color": "#ADAAAA"},
                                    name = "B"
                                )
                            ],
                            'layout': go.Layout(
                                annotations = [
                                    {
                                      "x": -0.0111111111111,
                                      "y": 2381.92771084,
                                      "font": {
                                        "color": "rgb(0, 0, 0)",
                                        "family": "Raleway",
                                        "size": 10
                                      },
                                      "showarrow": False,
                                      "text": "$2,242",
                                      "xref": "x",
                                      "yref": "y"
                                    },
                                    {
                                      "x": 0.995555555556,
                                      "y": 509.638554217,
                                      "font": {
                                        "color": "rgb(0, 0, 0)",
                                        "family": "Raleway",
                                        "size": 10
                                      },
                                      "showarrow": False,
                                      "text": "$329",
                                      "xref": "x",
                                      "yref": "y"
                                    },
                                    {
                                      "x": 0.995551020408,
                                      "y": 1730.32432432,
                                      "font": {
                                        "color": "rgb(0, 0, 0)",
                                        "family": "Raleway",
                                        "size": 10
                                      },
                                      "showarrow": False,
                                      "text": "You save<br><b>$1,913</b>",
                                      "xref": "x",
                                      "yref": "y"
                                    }
                                  ],
                                  autosize = False,
                                  height = 150,
                                  width = 340,
                                  bargap = 0.4,
                                  barmode = "stack",
                                  hovermode = "closest",
                                  margin = {
                                    "r": 40,
                                    "t": 20,
                                    "b": 20,
                                    "l": 40
                                  },
                                  showlegend = False,
                                  title = "",
                                  xaxis = {
                                    "autorange": True,
                                    "range": [-0.5, 1.5],
                                    "showline": True,
                                    "tickfont": {
                                      "family": "Raleway",
                                      "size": 10
                                    },
                                    "title": "",
                                    "type": "category",
                                    "zeroline": False
                                  },
                                  yaxis = {
                                    "autorange": False,
                                    "mirror": False,
                                    "nticks": 3,
                                    "range": [0, 3000],
                                    "showgrid": True,
                                    "showline": True,
                                    "tickfont": {
                                      "family": "Raleway",
                                      "size": 10
                                    },
                                    "tickprefix": "$",
                                    "title": "",
                                    "type": "linear",
                                    "zeroline": False
                                  }
                            )
                        },
                        config={
                            'displayModeBar': False
                        }
                    )
                ], className="six columns"),

            ], className="row "),

            # Row 3

            html.Div([

                html.Div([
                    html.H6(["Fees"],
                            className="gs-header gs-table-header padded"),

                    html.Br([]),

                    html.Div([

                        html.Div([
                            html.Strong(["Purchase fee"])
                        ], className="three columns right-aligned"),

                        html.Div([
                            html.P(["None"])
                        ], className="nine columns")


                    ], className="row "),

                    html.Div([

                        html.Div([
                            html.Strong(["Redemption fee"])
                        ], className="three columns right-aligned"),

                        html.Div([
                            html.P(["None"])
                        ], className="nine columns")

                    ], className="row "),

                    html.Div([

                        html.Div([
                            html.Strong(["12b-1 fee"])
                        ], className="three columns right-aligned"),

                        html.Div([
                            html.P(["None"])
                        ], className="nine columns")

                    ], className="row "),

                    html.Div([

                        html.Div([
                            html.Strong(["Account service fee"])
                        ], className="three columns right-aligned"),

                        html.Div([
                            html.Strong(["Nonretirement accounts, traditional IRAs, Roth IRAs, UGMAs/UTMAs, SEP-IRAs, and education savings accounts (ESAs)"]),
                            html.P(["We charge a $20 annual account service fee for each Vanguard Brokerage Account, as well as each individual Vanguard mutual fund holding with a balance of less than $10,000 in an account. This fee does not apply if you sign up for account access on vanguard.com and choose electronic delivery of statements, confirmations, and Vanguard fund reports and prospectuses. This fee also does not apply to members of Flagship Select™, Flagship®, Voyager Select®, and Voyager® Services."]),
                            html.Br([]),
                            html.Strong(["SIMPLE IRAs"]),
                            html.P(["We charge participants a $25 annual account service fee for each fund they hold in their Vanguard SIMPLE IRA. This fee does not apply to members of Flagship Select, Flagship, Voyager Select, and Voyager Services."]),
                            html.Br([]),
                            html.Strong(["403(b)(7) plans"]),
                            html.P(["We charge participants a $15 annual account service fee for each fund they hold in their Vanguard 403(b)(7) account. This fee does not apply to members of Flagship Select, Flagship, Voyager Select, and Voyager Services."]),
                            html.Br([]),
                            html.Strong(["Individual 401(k) plans"]),
                            html.P(["We charge participants a $20 annual account service fee for each fund they hold in their Vanguard Individual 401(k) account. This fee will be waived for all participants in the plan if at least 1 participant qualifies for Flagship Select, Flagship, Voyager Select, and Voyager Services"]),
                            html.Br([]),
                        ], className="nine columns")

                    ], className="row ")

                ], className="twelve columns")

            ], className="row "),

        ], className="subpage")

    ], className="page")

SVM = html.Div([  # page 5

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
                    html.H6(["Distributions"],
                            className="gs-header gs-table-header padded"),
                    html.Strong(["Distributions for this fund are scheduled quaterly"])
                ], className="twelve columns"),

            ], className="row "),

            # Row 2

            html.Div([

                html.Div([
                    html.Br([]),
                    html.H6(["Dividend and capital gains distributions"], className="gs-header gs-table-header tiny-header"),
                    html.Table(make_dash_table(df_dividend), className="tiny-header")
                ], className="twelve columns"),

            ], className="row "),

            # Row 3

            html.Div([

                html.Div([
                    html.H6(["Realized/unrealized gains as of 01/31/2018"], className="gs-header gs-table-header tiny-header")
                ], className=" twelve columns")

            ], className="row "),

            # Row 4

            html.Div([

                html.Div([
                    html.Table(make_dash_table(df_realized))
                ], className="six columns"),

                html.Div([
                    html.Table(make_dash_table(df_unrealized))
                ], className="six columns"),

            ], className="row "),

        ], className="subpage")

    ], className="page")

takeaways = html.Div([  # page 6

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
                    html.H6('Vanguard News',
                            className="gs-header gs-text-header padded"),
                    html.Br([]),
                    html.P('10/25/16    The rise of indexing and the fall of costs'),
                    html.Br([]),
                    html.P("08/31/16    It's the index mutual fund's 40th anniversary: Let the low-cost, passive party begin")
                ], className="six columns"),

                html.Div([
                    html.H6("Reviews",
                            className="gs-header gs-table-header padded"),
                    html.Br([]),
                    html.Li('Launched in 1976.'),
                    html.Li('On average, has historically produced returns that have far outpaced the rate of inflation.*'),
                    html.Li("Vanguard Quantitative Equity Group, the fund's advisor, is among the world's largest equity index managers."),
                    html.Br([]),
                    html.P("Did you know? The fund launched in 1976 as Vanguard First Index Investment Trust—the nation's first index fund available to individual investors."),
                    html.Br([]),
                    html.P("* The performance of an index is not an exact representation of any particular investment, as you cannot invest directly in an index."),
                    html.Br([]),
                    html.P("Past performance is no guarantee of future returns. See performance data current to the most recent month-end.")
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
