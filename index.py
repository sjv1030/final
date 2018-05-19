import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import app_ng, app_oil


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/app_ng/overview':
        return app_ng.overview
    elif pathname == '/apps/app_ng/multivariable_ARIMA':
        return app_ng.multivariable_ARIMA
    elif pathname == '/apps/app_ng/fb_prophet':
        return app_ng.fbProphet
    elif pathname == '/apps/app_ng/RNN':
        return app_ng.RNN
    elif pathname == '/apps/app_ng/SVM':
        return app_ng.SVM
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
    else:
        "404"
if __name__ == '__main__':
    app.run_server(debug=True)
