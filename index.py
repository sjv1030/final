import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import app_ng, app_oil

"""
@author: michelebradley
"""

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page = html.Div([
    dcc.Link('Natural Gas Anaysis', href='/apps/app_ng/overview'),
    html.Br(),
    dcc.Link('Oil Analysis', href='/apps/app_oil/overview'),
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/app_ng/overview':
        return app_ng.overview
    elif pathname == '/apps/app_ng/multivariable':
        return app_ng.multivariable
    elif pathname == '/apps/app_ng/fb_prophet':
        return app_ng.fbProphet
    elif pathname == '/apps/app_ng/LSTM':
        return app_ng.LSTM
    elif pathname == '/apps/app_ng/takeaways':
        return app_ng.takeaways
    elif pathname == '/apps/app_oil/overview':
        return app_oil.overview
    elif pathname == '/apps/app_oil/price-performance':
        return app_oil.pricePerformance
    elif pathname == '/apps/app_oil/portfolio-management':
            return app_oil.portfolioManagement
    elif pathname == '/apps/app_oil/fees':
        return app_oil.feesMins
    elif pathname == '/apps/app_oil/distributions':
        return app_oil.distributions
    elif pathname == '/apps/app_oil/news-and-reviews':
        return app_oil.newsReviews
    elif pathname == "/":
        index_page
    else:
        "404"


if __name__ == '__main__':
    app.run_server(debug=True)
