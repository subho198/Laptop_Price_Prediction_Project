from scipy import rand
import streamlit as st
import sklearn
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder

rf_clf=pickle.load(open('random_forest.pkl','rb'))
ndf=pickle.load(open('final_df.pkl','rb'))


st.title('Laptop Price Predictor App by Subhajit')

col1,col2=st.columns(2)

with col1:
    Company=st.selectbox("Select the Brand Name : ",ndf['Company'].unique())
with col2:
    TypeName=st.selectbox("Select Product Type : ",ndf['TypeName'].unique())

col3,col4=st.columns(2)

with col3:
    Cpu_Vender=st.selectbox("Select Processor Brand : ",ndf['Cpu_Vender'].unique())
with col4:
    Cpu_Type=st.selectbox("Select Processor Type : ",ndf['Cpu_Type'].unique())

col5,col6=st.columns(2)

with col5:
    Storage_Type=st.selectbox("Select Storage Type : ",ndf['Storage Type'].unique())
with col6:
    Gpu_Vender=st.selectbox("Select GPU Brand : ",ndf['Gpu_Vender'].unique())

col7,col8=st.columns(2)
with col7:
    Gpu_Type=st.selectbox("Select GPU Type : ",ndf['Gpu_Type'].unique())
with col8:
    OpSys=st.selectbox("Select The Operating System : ",ndf['OpSys'].unique())

col9,col10,col11=st.columns(3)

with col9:
    Inches=st.number_input('Enter the Screen Size (in Inches) : ')

with col10:
    Touchscreen=st.selectbox('Is it Touchscreen?',ndf['Touchscreen'].unique())

with col11:
    Ips=st.selectbox('Is it Ips Display?',ndf['Ips'].unique())

col12,col13,col14=st.columns(3)

with col12:
    Ram=st.number_input('RAM size : ')

with col13:
    Storage_GB=st.number_input('Storage size in GB: ')

with col14:
    Weight=st.number_input('Weight in Kg : ')

if st.button("Predict Laptop's Price :"):
    input_df=pd.DataFrame({'Company':[Company], 'TypeName':[TypeName], 'Inches':[Inches], 'Touchscreen':[Touchscreen], 'Ips':[Ips],'Cpu_Vender':[Cpu_Vender],'Cpu_Type':[Cpu_Type], 'Ram':[Ram],'Storage (GB)':[Storage_GB],'Storage Type':[Storage_Type],'Gpu_Vender':[Gpu_Vender],'Gpu_Type':[Gpu_Type],'Weight':[Weight],'OpSys':[OpSys]})
    st.table(input_df)
    result=rf_clf.predict(input_df)
    st.subheader(f"The approximate price of the Laptop is : {result} ")