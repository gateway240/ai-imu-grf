# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 09:14:28 2024

@author: akupeki
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate

# Reading 3D accelerometer/gyro values 
def convert_strings_to_floats_acc(input_array):
    X_out=[]
    Y_out=[]
    Z_out=[]
    for element in input_array:
        x,y,z = element.split(',')    
        X_out.append(float(x))
        Y_out.append(float(y))
        Z_out.append(float(z))
    return np.array(X_out), np.array(Y_out), np.array(Z_out)  

# Reading quaterion values
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

#exclude these ID's from the data set
excluded_IDs= ['11', '14', '37', '49']

#necessary headers for data  
imu_loc_names= ['pelvis_imu','tibia_r_imu','femur_r_imu','tibia_l_imu','femur_l_imu','calcn_r_imu','calcn_l_imu']
grf_channel_names = ['f1_1', 'f1_2', 'f1_3']

#directory path
dir_path = "C:\\Users\\akupeki\\Documents\\kuopio-gait-dataset-processed-v2\\"    

#get ID list
ID_list = os.listdir(dir_path)
#exclude unwanted ID's
ID_list = [ID for ID in ID_list if ID not in excluded_IDs]
    
#start loop for all ID's
for ID in ID_list:
    #extend the path
    dir_path_imu = dir_path + ID + "\\imu_extracted\\" 
    dir_path_grf = dir_path + ID + "\\mocap\\"
    
    #filter the filenames in imu directory:
    filenames = os.listdir(dir_path_imu)
    imu_files = [file for file in filenames if file.endswith(".sto")]
    
    #filter the filenames in mocap directory:
    filenames = os.listdir(dir_path_grf)

    substring1 = "calib"
    substring2 = "mass"

    grf_files = [file for file in filenames if file.endswith("grfs.sto")]
    grf_files = [file for file in grf_files if substring1 not in file]
    grf_files = [file for file in grf_files if substring2 not in file]
    



    
    
dir_path_imu = dir_path + ID_list[0] + "\\imu_extracted\\"    
f_name_imu = dir_path_imu + "data_l_comf_01_00B42D4D.txt"
f_name_acc = dir_path_imu + "data_l_comf_01_accelerations.sto" 
f_name_ori = dir_path_imu + "data_l_comf_01_orientations.sto"

dir_path_grf = dir_path + ID_list[0] + "\\mocap\\" 
f_name_grf = dir_path_grf + "l_comf_01_grfs.sto"


#filter the filenames in imu directory:
filenames = os.listdir(dir_path_imu)
imu_files = [file for file in filenames if file.endswith(".sto")]


#filter the filenames in mocap directory:
filenames = os.listdir(dir_path_grf)

substring1 = "calib"
substring2 = "mass"

grf_files = [file for file in filenames if file.endswith("grfs.sto")]
grf_files = [file for file in grf_files if substring1 not in file]
grf_files = [file for file in grf_files if substring2 not in file]

# acc_channel_names= ['Acc_X','Acc_Y','Acc_Z']
# acc_x = df_imu[acc_channel_names[0]].values
# acc_y = df_imu[acc_channel_names[1]].values
# acc_z = df_imu[acc_channel_names[2]].values

df_acc = pd.read_csv(f_name_acc,skiprows=[1,2,3,4],header=1,delimiter='\t')
df_ori = pd.read_csv(f_name_ori,skiprows=[1,2,3,4],header=1,delimiter='\t')
df_grf = pd.read_csv(f_name_grf,skiprows=[1,2,3,4],header=1,delimiter='\t')



def extract_and_window_data(file_path,channel_names,window_size):
    df = pd.read_csv(file_path,skiprows=[1,2,3,4],header=1,delimiter='\t')
    
    

