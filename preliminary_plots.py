# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 09:14:28 2024

@author: akupeki
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate

dir_path = ""

file_name_grf = "l_comf_01_grfs.sto"
file_name_ori = "l_comf_01-000_orientations.sto"
file_name_gyr = "l_comf_01-000_gyros.sto"

def convert_strings_to_floats_gyr(input_array):
    X_out=[]
    Y_out=[]
    Z_out=[]
    for element in input_array:
        x,y,z = element.split(',')    
        X_out.append(float(x))
        Y_out.append(float(y))
        Z_out.append(float(z))
    return np.array(X_out), np.array(Y_out), np.array(Z_out)  

def convert_strings_to_floats_ori(input_array):
    q1_out =[] 
    q2_out =[] 
    q3_out =[] 
    q4_out =[] 
    for element in input_array:
        q1,q2,q3,q4 = element.split(',')    
        q1_out.append(float(q1))
        q2_out.append(float(q2))
        q3_out.append(float(q3))
        q4_out.append(float(q4))
    return np.array(q1_out), np.array(q2_out), np.array(q3_out), np.array(q4_out)  

#sampling frequency
fs = 100
  
#read csv files to dataframes  
df_grf = pd.read_csv(dir_path + file_name_grf,skiprows=[1,2,3,4],header=1,delimiter='\t')
df_ori = pd.read_csv(dir_path + file_name_ori,skiprows=[1,2,3,4],header=1,delimiter='\t')
df_gyr = pd.read_csv(dir_path + file_name_gyr,skiprows=[1,2,3,4],header=1,delimiter='\t')

#extract channels ö
gyro_pelvis = df_gyr['pelvis_imu'].values
ori_pelvis = df_ori['pelvis_imu'].values
grf_x = df_grf['f1_1']
grf_y = df_grf['f1_2']
grf_z = df_grf['f1_3']

#downsampling factor
decim_factor = int(1000/fs)

grf_x = decimate(grf_x,decim_factor)
grf_y = decimate(grf_y,decim_factor)
grf_z = decimate(grf_z,decim_factor) 

#time vectors
time = np.linspace(0, (len(df_gyr)-1)/fs, len(df_gyr))
time_grf = np.linspace(0, (len(grf_z)-1)/fs, len(grf_z))

#extract quaternion and gyro data from a string
gyro_pelvis_x,gyro_pelvis_y,gyro_pelvis_z= convert_strings_to_floats_gyr(gyro_pelvis)
ori_pelvis_q1,ori_pelvis_q2,ori_pelvis_q3,ori_pelvis_q4 = convert_strings_to_floats_ori(ori_pelvis)

#use the prominent component of force to determine activation of the force plate
index_z = np.where(np.abs(grf_z)>5)

#extended window length
indices= index_z[0]
window = np.arange(indices[0]-len(grf_z[index_z]),indices[len(indices)-1]+len(grf_z[index_z]),1)

# this plot is for the segment where force plate is struck.
fig,ax = plt.subplots(3,figsize=(20,50))
ax[0].plot(time[index_z],gyro_pelvis_x[index_z])
ax[0].plot(time[index_z],gyro_pelvis_y[index_z])
ax[0].plot(time[index_z],gyro_pelvis_z[index_z])

ax[1].plot(time[index_z],ori_pelvis_q1[index_z])
ax[1].plot(time[index_z],ori_pelvis_q2[index_z])
ax[1].plot(time[index_z],ori_pelvis_q3[index_z])
ax[1].plot(time[index_z],ori_pelvis_q4[index_z])

ax[2].plot(time_grf[index_z],grf_x[index_z])
ax[2].plot(time_grf[index_z],grf_y[index_z])
ax[2].plot(time_grf[index_z],grf_z[index_z])

#this plot has wider time window, 3 times the segment lenght
# fig,ax = plt.subplots(3,figsize=(20,50))
# ax[0].plot(time[window],gyro_pelvis_x[window])
# ax[0].plot(time[window],gyro_pelvis_y[window])
# ax[0].plot(time[window],gyro_pelvis_z[window])

# ax[1].plot(time[window],ori_pelvis_q1[window])
# ax[1].plot(time[window],ori_pelvis_q2[window])
# ax[1].plot(time[window],ori_pelvis_q3[window])
# ax[1].plot(time[window],ori_pelvis_q4[window])

# ax[2].plot(time_grf[window],grf_x[window])
# ax[2].plot(time_grf[window],grf_y[window])
# ax[2].plot(time_grf[window],grf_z[window])

plt.show()

#for saving the results
#plt.savefig(dir_path + 'gyr_ori_grf_decimated_window_extended.pdf',format = 'pdf', bbox_inches = 'tight')


