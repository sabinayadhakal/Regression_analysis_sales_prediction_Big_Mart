import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

with open('sales_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

feature_columns = [
    'Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year',
    'Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type',
    'Item_Identifier', 'Item_Type', 'Outlet_Identifier'
]

fat_options = ['Low Fat', 'Regular']
size_options = ['Small', 'Medium', 'High']
location_options = ['Tier 1', 'Tier 2', 'Tier 3']
type_options = ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3']
item_types = ['Fruits and Vegetables', 'Snack Foods', 'Household', 'Frozen Foods']  
outlet_ids = ['OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027'] 
item_ids = ['FDA15', 'DRC01', 'FDN15', 'FDX07']  

st.title("🛒 BigMart Sales Prediction App")

st.sidebar.header("Enter Product Information")

user_input = {
    'Item_Identifier': st.sidebar.selectbox("Item Identifier", item_ids),
    'Item_Weight': st.sidebar.number_input("Item Weight", min_value=0.0, step=0.1),
    'Item_Fat_Content': st.sidebar.selectbox("Fat Content", fat_options),
    'Item_Visibility': st.sidebar.slider("Item Visibility", 0.0, 0.3, 0.05),
    'Item_Type': st.sidebar.selectbox("Item Type", item_types),
    'Item_MRP': st.sidebar.number_input("Item MRP", min_value=0.0, step=1.0),
    'Outlet_Identifier': st.sidebar.selectbox("Outlet ID", outlet_ids),
    'Outlet_Establishment_Year': st.sidebar.number_input("Establishment Year", min_value=1985, max_value=2025),
    'Outlet_Size': st.sidebar.selectbox("Outlet Size", size_options),
    'Outlet_Location_Type': st.sidebar.selectbox("Outlet Location", location_options),
    'Outlet_Type': st.sidebar.selectbox("Outlet Type", type_options)
}

input_df = pd.DataFrame([user_input])

input_df['Item_Fat_Content'] = input_df['Item_Fat_Content'].replace({'Low Fat': 0, 'Regular': 1})
input_df['Outlet_Size'] = input_df['Outlet_Size'].replace({'Small': 0, 'Medium': 1, 'High': 2})
input_df['Outlet_Location_Type'] = input_df['Outlet_Location_Type'].replace({'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2})
input_df['Outlet_Type'] = input_df['Outlet_Type'].replace({
    'Grocery Store': 0,
    'Supermarket Type1': 1,
    'Supermarket Type2': 2,
    'Supermarket Type3': 3
})

item_type_encoded = pd.get_dummies(input_df['Item_Type'], prefix='Item_Type')
input_df = pd.concat([input_df.drop('Item_Type', axis=1), item_type_encoded], axis=1)

outlet_encoded = pd.get_dummies(input_df['Outlet_Identifier'], prefix='Outlet')
input_df = pd.concat([input_df.drop('Outlet_Identifier', axis=1), outlet_encoded], axis=1)

item_id_encoded = pd.get_dummies(input_df['Item_Identifier'], prefix='Item_ID')
input_df = pd.concat([input_df.drop('Item_Identifier', axis=1), item_id_encoded], axis=1)

for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model.feature_names_in_]

# Predict
if st.button("Predict Sales"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Sales: ₹{prediction:,.2f}")
