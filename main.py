import streamlit as st
import pandas as pd
import numpy as np
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_cross_validation_metric
import base64
#import neuralprophet as nprophet
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.plot import plot_plotly
import plotly.offline as py
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.plot import plot_components_plotly
st.title('پیش‌بینی اتوماتیک با استفاده از پکیج پیامبر فیس‌بوک')


today = dt.date.today()

before = today - dt.timedelta(days=7)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)

if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')

st.title("Forcaster")

function_list = ['fbprophet', 'Neural Networks']
sidebar_function = st.sidebar.selectbox("Choose the forecasting method", function_list)
crypotocurrencies = (
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SAND-USD', 'MANA-USD', 'XRP-USD', 'LTC-USD', 'EOS-USD', 'XLM-USD',
    'TRX-USD', 'ETC-USD', 'SHIB-USD', 'DOGE-USD', 'TRX-USD', 'SOL-USD', 'FTM-USD', 'MATIC-USD',)

selected_stock = st.selectbox('Select dataset for prediction', crypotocurrencies)

n_years = st.slider('Hours of prediction:', 1, 24)
period = n_years / 24


@st.cache
def load_data(ticker):
    data = yf.download(ticker, start_date, end_date, interval='1min')
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
fig = px.line(data, x='Date', y='Close')
st.plotly_chart(fig)



"""
### Step 2: Select Forecast Horizon

Keep in mind that forecasts become less accurate with larger forecast horizons.
"""

periods_input = st.number_input('How many periods would you like to forecast into the future?',
                                min_value=1, max_value=365)

if df is not None:
    m = Prophet(seasonality_mode='multiplicative', seasonality_prior_scale=5)
    #m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    #m.add_country_holidays(country_name='US')
    m.fit(data)

"""
### Step 3: Visualize Forecast Data

The below visual shows future predicted values. "yhat" is the predicted value, and the upper and lower limits are (by default) 80% confidence intervals.
"""
if df is not None:
    future = m.make_future_dataframe(periods=periods_input)

    forecast = m.predict(future)
    fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    fcst_filtered = fcst[fcst['ds'] > max_date]
    st.write(fcst_filtered)

    """
    The next visual shows the actual (black dots) and predicted (blue line) values over time.
    """
    fig = plot_plotly(m, forecast, trend=True)  # This returns a plotly Figure
    st.write(fig)

    """
    The next few visuals show a high level trend of predicted values, day of week trends, and yearly trends (if dataset covers multiple years). The blue shaded area represents upper and lower confidence intervals.
    """
    fig2 = plot_components_plotly(m, forecast,
                                 figsize=(800, 175))
    st.write(fig2)

"""
### Step 4: Download the Forecast Data

The below link allows you to download the newly created forecast to your computer for further analysis and use.
"""
if df is not None:
    csv_exp = fcst_filtered.to_csv(index=False)
    # When no file name is given, pandas returns the CSV as a string, nice.
    b64 = base64.b64encode(csv_exp.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as ** &lt;forecast_name&gt;.csv**)'
    st.markdown(href, unsafe_allow_html=True)
