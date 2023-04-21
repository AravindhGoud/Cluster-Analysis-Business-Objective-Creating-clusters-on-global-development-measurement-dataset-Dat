

import streamlit as st
st.title('Cluster Prediction Model For Global Development Measurement Dataset')

# -*- coding: utf-8 -*-
"""
Author @Aravindh Goud S & Team
"""

import streamlit as st 
import pickle
import numpy as np

with open(r"C:\Users\aravi\OneDrive\Desktop\Project Files\Deployment\trained_model_clustering.pkl", "rb") as f:
    loaded_model = pickle.load(f, encoding="utf-8")

scaler = loaded_model['scaler']
pca = loaded_model['pca']
kmean = loaded_model['kmean']
# write the function for scaling and pca
def prediction(input_data):
    
    #changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance 
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        
    input_scaled = scaler.transform(input_data_reshaped)
    
    input_pca = pca.transform(input_scaled)
        
    predict =kmean.predict(input_pca)
     
            
    if predict[0] == 2:
        return 'Developed country'
    elif predict[0] == 1:
        return 'Under Developed Country'
    else:
        return 'Developing Country'
    



# Define the Streamlit app
def main():
    # Set the app title
    
    # Getting inout from user
    Country = st.sidebar.text_input('Enter Country Name')
    Birth_Rate = st.sidebar.number_input('Enter the Birth Rate ')
    CO2_Emissions = st.sidebar.number_input('Enter the CO2 Emissions ')
    Energy_Usage = st.sidebar.number_input('Enter the Energy Usage')
    GDP = st.sidebar.number_input('Enter the GDP')
    Health_ExpGDP= st.sidebar.number_input('Enter the Health Expenditure % on GDP')
    Infant_Mortality_Rate = st.sidebar.number_input("Enter the Infant Mortality Rate (0-1)",min_value=0.0,max_value=1.0,step=0.001)
    Internet_Usage = st.sidebar.number_input("Enter the Internet Usage  (0-1)",min_value=0.0,max_value=1.0,step=0.001)
    LifeExpFemale = st.sidebar.number_input('Enter the Life Expectancy of Female')
    LifeExpMale = st.sidebar.number_input('Enter the Life Expectancy of Male')
    MobilePhoneUsage = st.sidebar.number_input("Enter the Mobile Phone Usage (0-1)",min_value=0.0,max_value=1.0,step=0.01)
    Population_0to14 = st.sidebar.number_input("Enter the Value of % Population of Age between 0-14",min_value=0.0,max_value=1.0,step=0.01)
    Population_15to64 = st.sidebar.number_input("Enter the value of % Population of Age between 15-64",min_value=0.0,max_value=1.0,step=0.01)
    Population_65plus = st.sidebar.number_input("Enter the value of % Population of Age above 65",min_value=0.0,max_value=1.0,step=0.01)
    Population_Total = st.sidebar.number_input("Enter the Total Population Count")
    UrbanPopulation = st.sidebar.number_input("Enter the Value of % Population in Urban")
    Tourism_inbound = st.sidebar.number_input("Enter the Tourism Inbound")
    Tourism_outbound     = st.sidebar.number_input("Enter the Tourism Outbound")

   

    if st.button('Clustering Prdiction Result'):
       pred = prediction([Birth_Rate, CO2_Emissions,Energy_Usage,GDP,Health_ExpGDP,Infant_Mortality_Rate,Internet_Usage,LifeExpFemale,LifeExpMale,MobilePhoneUsage,Population_0to14,Population_15to64,Population_65plus,Population_Total,UrbanPopulation,Tourism_inbound,Tourism_outbound])
       st.success(pred)
    
# Run the app
if __name__ == '__main__':
    main()
