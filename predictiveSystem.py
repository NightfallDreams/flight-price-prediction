# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the model
loaded_model = pickle.load(open('C:/Users/OneDrive/Desktop/FLIGHTPRICEML/rf_random_model.pkl', 'rb'))

# Sample input data
input_data = {
    'Airline': ['IndiGo'],
    'Total_Stops': ['non-stop'],
    'Route1': ['BLR'],
    'Route2': ['DEL'],
    'Route3': ['None'],
    'Route4': ['None'],
    'Route5': ['None'],
    'Duration_hours': [2],
    'Duration_min': [50],
    'journey_day': [24],
    'journey_month': [3],
    'Dep_Time_hour': [22],
    'Dep_Time_min': [20],
    'Arrival_Time_hour': [1],
    'Arrival_Time_min': [10],
    'Source': ['Banglore'],
    'Destination': ['New Delhi']
}

# Convert the input data into a DataFrame
input_df = pd.DataFrame(input_data)

# Initialize LabelEncoder
le = LabelEncoder()

# Categorical columns to encode
categorical_cols = ['Airline', 'Total_Stops', 'Route1', 'Route2', 'Route3', 'Route4', 'Route5', 'Source', 'Destination']

# Apply LabelEncoder to categorical columns
for col in categorical_cols:
    input_df[col] = le.fit_transform(input_df[col])

# One-hot encode 'Airline', 'Source', and 'Destination'
Airline = pd.get_dummies(input_df['Airline'], drop_first=True)
source = pd.get_dummies(input_df['Source'], drop_first=True)
destination = pd.get_dummies(input_df['Destination'], drop_first=True)

# Drop the original categorical columns
input_df.drop(['Airline', 'Source', 'Destination'], axis=1, inplace=True)

# Concatenate the one-hot encoded columns to the DataFrame
final_input_df = pd.concat([input_df, Airline, source, destination], axis=1)

# Align the features of input data with the model's expected columns
model_columns = loaded_model.best_estimator_.feature_names_in_
final_input_df = final_input_df.reindex(columns=model_columns, fill_value=0)

# Make prediction using the model
prediction = loaded_model.predict(final_input_df)

# Print the predicted price
print(f"Predicted Price: {prediction[0]}")
