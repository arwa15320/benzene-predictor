import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

st.set_page_config(page_title='Benzene Predictor', layout='wide')

@st.cache_resource
def load_model():
    with open('benzene_model.pkl', 'rb') as f:
        return pickle.load(f)

package  = load_model()
model    = package['model']
scaler   = package['scaler']
features = package['features']

st.title('Air Quality — Benzene Concentration Predictor')
st.markdown('**CSAI-801 | Group 17 | Mohamed Mahmoud Abdel Majid & Arwa Elgazar**')
st.markdown(
    'This application predicts the concentration of Benzene (C6H6) '
    'using only low-cost metal oxide sensor readings.'
)
st.divider()

# Sidebar — date and time
st.sidebar.header('Date and Time')
selected_date = st.sidebar.date_input('Date', value=datetime(2025, 3, 1))
selected_hour = st.sidebar.slider('Hour of Day', 0, 23, 9)

month      = selected_date.month
dow        = selected_date.weekday()
is_weekend = int(dow >= 5)
season     = {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}[month]

day_label    = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][dow]
season_label = ['Winter','Spring','Summer','Autumn'][season]

st.sidebar.markdown('---')
st.sidebar.markdown(f'**Day:** {day_label}')
st.sidebar.markdown(f'**Season:** {season_label}')
st.sidebar.markdown(f'**Weekend:** {"Yes" if is_weekend else "No"}')

# Sensor inputs
st.subheader('Sensor Readings')
st.caption('Default values represent typical hourly readings from the dataset.')

col1, col2 = st.columns(2)

with col1:
    st.markdown('**Metal Oxide Sensor Signals**')
    pt08_s1 = st.number_input(
        'PT08.S1 — CO Sensor',
        min_value=600.0, max_value=2100.0, value=1063.0, step=10.0,
        help='Typical range: 647 to 2040'
    )
    pt08_s3 = st.number_input(
        'PT08.S3 — NOx Sensor',
        min_value=300.0, max_value=2700.0, value=835.0, step=10.0,
        help='Typical range: 322 to 2683'
    )
    pt08_s4 = st.number_input(
        'PT08.S4 — NO2 Sensor',
        min_value=500.0, max_value=2800.0, value=1446.0, step=10.0,
        help='Typical range: 551 to 2775'
    )
    pt08_s5 = st.number_input(
        'PT08.S5 — O3 Sensor',
        min_value=200.0, max_value=2600.0, value=963.0, step=10.0,
        help='Typical range: 221 to 2523'
    )

with col2:
    st.markdown('**Weather Conditions**')
    temp = st.slider('Temperature (C)', -5.0, 45.0, 17.8, step=0.5,
                     help='Dataset range: -1.9 to 44.6')
    rh   = st.slider('Relative Humidity (%)', 9.0, 89.0, 49.3, step=1.0,
                     help='Dataset range: 9.2 to 88.7')
    ah   = st.slider('Absolute Humidity (g/m3)', 0.18, 2.24, 0.978, step=0.01,
                     help='Dataset range: 0.185 to 2.231')

# Build input in exact feature order
input_dict = {
    'PT08.S1(CO)'  : pt08_s1,
    'PT08.S3(NOx)' : pt08_s3,
    'PT08.S4(NO2)' : pt08_s4,
    'PT08.S5(O3)'  : pt08_s5,
    'T'            : temp,
    'RH'           : rh,
    'AH'           : ah,
    'Hour'         : selected_hour,
    'DayOfWeek'    : dow,
    'Month'        : month,
    'IsWeekend'    : is_weekend,
    'Season'       : season,
}

input_df     = pd.DataFrame([input_dict])[features]
input_scaled = scaler.transform(input_df)

st.divider()
if st.button('Predict Benzene Concentration', type='primary',
             use_container_width=True):

    prediction = float(model.predict(input_scaled)[0])
    prediction = max(0.0, prediction)

    if prediction < 5.0:
        level  = 'Low'
        color  = 'green'
        advice = 'Benzene levels are within an acceptable range.'
    elif prediction < 15.0:
        level  = 'Moderate'
        color  = 'orange'
        advice = 'Elevated benzene detected. Ventilation is recommended.'
    else:
        level  = 'High — Health Risk'
        color  = 'red'
        advice = 'Dangerous benzene levels detected. Limit outdoor exposure.'

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric('Predicted C6H6 (Benzene)', f'{prediction:.2f} ug/m3')

    with c2:
        st.markdown('**Risk Level**')
        st.markdown(
            f'<h3 style="color:{color}">{level}</h3>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown('**WHO Annual Guideline**')
        st.info('Recommended limit: 1.7 ug/m3')

    st.markdown(f'**Advice:** {advice}')

    st.markdown('**How does this compare to the dataset?**')
    pct = min(int(prediction / 63.74 * 100), 100)
    st.progress(pct)
    st.caption(
        f'Your prediction: {prediction:.2f}  |  '
        f'Dataset mean: 10.08  |  '
        f'Dataset max: 63.74  (all values in ug/m3)'
    )

    with st.expander('View full input sent to model'):
        st.dataframe(input_df.T.rename(columns={0: 'Value'}))

    with st.expander('About this model'):
        st.markdown('''
        **Model:** Support Vector Regression (RBF kernel)

        **Training data:** UCI Air Quality Dataset — 8,991 hourly readings
        collected between March 2004 and April 2005

        **Features used:** 12 features — four metal oxide sensor signals,
        three weather measurements, and five temporal features

        **Features removed:** CO(GT), NOx(GT), NO2(GT) are ground truth
        values measured by a certified reference analyzer and would not
        be available in a real low-cost deployment. PT08.S2(NMHC) was
        removed due to a 0.98 correlation with the target variable.

        **Test R2 Score:** 0.9228

        **Test MAE:** 1.31 ug/m3
        ''')
