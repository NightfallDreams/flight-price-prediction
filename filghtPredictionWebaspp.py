import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from datetime import datetime

# Set background image (use an image URL or upload it)
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://as1.ftcdn.net/v2/jpg/05/22/26/36/1000_F_522263674_pbveyKlSHuvuFNIh5CLhrKVlg5FS1SuV.jpg'); /* Use a proper image URL */
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .profile-section {
        background-color: rgba(0, 0, 0, 0.5);
        padding: 10px;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .profile-section img {
        border-radius: 50%;
        width: 100px;
        height: 100px;
    }
    .input-container {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 15px;
        border-radius: 10px;
    }
    .input-container input, .input-container select {
        margin-bottom: 10px;
        width: 100%;
        padding: 8px;
        border-radius: 5px;
    }
    .button {
        background-color: #5C6BC0;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        font-size: 16px;
    }
    .button:hover {
        background-color: #3F4B8B;
    }
    </style>
    """, unsafe_allow_html=True
)

# Profile Section
st.markdown(
    """
    <div class="profile-section">
        <img src="https://imgs.search.brave.com/-yQfCG0B6KqnGuzxvXw_M9kP4rfx3un-AeKOEPDYgc4/rs:fit:860:0:0:0/g:ce/aHR0cDovL20uZ2V0/dHl3YWxscGFwZXJz/LmNvbS93cC1jb250/ZW50L3VwbG9hZHMv/MjAyMy8wNy9QZnAt/U2F0b3J1LUdvam8u/anBn" alt="Profile Picture">  <!-- Use a proper image URL -->
        <h3>Welcome, Ashish!</h3>
        <p>Flight Price Prediction</p>
    </div>
    """, unsafe_allow_html=True
)

# Load the trained model
loaded_model = pickle.load(open('C:/Users/OneDrive/Desktop/FLIGHTPRICEML/rf_random_model.pkl', 'rb'))

# Define the flight price prediction function
def flightPricePrediction(input_data):
    input_df = pd.DataFrame(input_data)
    le = LabelEncoder()
    categorical_cols = ['Airline', 'Total_Stops', 'Route1', 'Route2', 'Route3', 'Route4', 'Route5', 'Source', 'Destination']

    for col in categorical_cols:
        input_df[col] = le.fit_transform(input_df[col])

    Airline = pd.get_dummies(input_df['Airline'], drop_first=True)
    source = pd.get_dummies(input_df['Source'], drop_first=True)
    destination = pd.get_dummies(input_df['Destination'], drop_first=True)

    input_df.drop(['Airline', 'Source', 'Destination'], axis=1, inplace=True)
    final_input_df = pd.concat([input_df, Airline, source, destination], axis=1)

    model_columns = loaded_model.best_estimator_.feature_names_in_
    final_input_df = final_input_df.reindex(columns=model_columns, fill_value=0)
    prediction = loaded_model.predict(final_input_df)

    return f"Predicted Price: {prediction[0]}"

# Streamlit app for user input
def main():
    st.markdown("<h2 style='color:white;'>Enter Flight Details</h2>", unsafe_allow_html=True)

    # Input fields
    with st.form(key='flight_form'):
        airline = st.selectbox("Select Airline", ['IndiGo', 'Air India', 'SpiceJet', 'GoAir', 'Vistara'])
        total_stops = st.selectbox("Select Total Stops", ['non-stop', '1 stop', '2 stops', '3 stops'])
        route1 = st.text_input("Route1 (e.g., BLR)")
        route2 = st.text_input("Route2 (e.g., DEL)")
        route3 = st.text_input("Route3 (if any, else None)")
        route4 = st.text_input("Route4 (if any, else None)")
        route5 = st.text_input("Route5 (if any, else None)")
        duration_hours = st.number_input("Duration Hours", min_value=0)
        duration_min = st.number_input("Duration Minutes", min_value=0)
        
        # Journey Date Input using calendar
        journey_date = st.date_input("Journey Date")
        
        # Departure Time Input using clock
        dep_time = st.time_input("Departure Time")
        
        # Arrival Time Input using clock
        arrival_time = st.time_input("Arrival Time")
        
        source = st.text_input("Source (e.g., Bangalore)")
        destination = st.text_input("Destination (e.g., New Delhi)")

        submit_button = st.form_submit_button("Predict Price")

        if submit_button:
            # Extracting the journey day, month, and year from the date input
            journey_day = journey_date.day
            journey_month = journey_date.month
            journey_year = journey_date.year

            # Extracting the hours and minutes for departure and arrival time
            dep_time_hour = dep_time.hour
            dep_time_min = dep_time.minute
            arrival_time_hour = arrival_time.hour
            arrival_time_min = arrival_time.minute

            input_data = {
                'Airline': [airline],
                'Total_Stops': [total_stops],
                'Route1': [route1],
                'Route2': [route2],
                'Route3': [route3],
                'Route4': [route4],
                'Route5': [route5],
                'Duration_hours': [duration_hours],
                'Duration_min': [duration_min],
                'journey_day': [journey_day],
                'journey_month': [journey_month],
                'journey_year': [journey_year],
                'Dep_Time_hour': [dep_time_hour],
                'Dep_Time_min': [dep_time_min],
                'Arrival_Time_hour': [arrival_time_hour],
                'Arrival_Time_min': [arrival_time_min],
                'Source': [source],
                'Destination': [destination]
            }

            result = flightPricePrediction(input_data)
            st.write(result)

if __name__ == "__main__":
    main()
