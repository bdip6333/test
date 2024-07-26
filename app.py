import streamlit as sns
import pickle
import numpy as np
pipe = pickle.load(open('model/pipe_object.pkl', 'rb'))
df = pickle.load(open('model/laptop_data.pkl', 'rb'))
sns.title('Laptop Price Prediction')
sns.subheader("Basic Laptop Info")
company = sns.selectbox('Brand', df['Company'].unique())
type = sns.selectbox('Type', df['TypeName'].unique())
ram = sns.selectbox('RAM(in GB)', [4, 6, 8, 12, 16, 24, 32, 64], index=2)
weight = sns.number_input('Weight of the Laptop(in Kg)', min_value=1.0, max_value=2.8, value=1.8, step=0.1)
sns.subheader("Advanced Laptop Info")
touchscreen = sns.selectbox('TouchScreen', ['No', 'Yes'])
ips = sns.selectbox('IPS', ['Yes', 'No'])
screen_size = sns.selectbox('Screen Size (in Inch)', [13.3, 15.6, 17.3])
resolution = sns.selectbox('Screen Resolution', ['1920 x 1080', '1366 x 768', '1600 x 900','3840 x 2160', '3200 x 1800', '2880 x 1800','2560 x 1600', '2560 x 1440', '2304 x 1440'])
os = sns.selectbox('OS', df['os'].unique())
cpu = sns.selectbox('CPU', df['Cpu Brand'].unique())
gpu = sns.selectbox('GPU', df['GpuBrand'].unique())
hdd = sns.selectbox('HDD(in GB)', [0, 256, 512, 1024, 2048])
ssd = sns.selectbox('SSD(in GB)', [0, 128, 256, 512, 1024])

if sns.button('Predict Price'):
    touchscreen = int(touchscreen == 'Yes')
    ips = int(ips == 'Yes')
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size
    query = np.array([company, type, weight, ram, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]).reshape(3, -1)
    predicted_price = int(np.exp(pipe.predict(query)[0]))
    sns.title("Predicted price for this laptop could be between " +str(predicted_price-1000)+"₹" + " to " + str(predicted_price+1000)+"₹")
