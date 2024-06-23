
import os
import sys
import cv2
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

csv_file = './data/Github_csv/TrainingData.csv'

sample_data = pd.read_csv(csv_file)

print('read all training file data!')

all_training_x_max = 0
all_training_x_min = 9999

all_training_y_max = 0
all_training_y_min = 9999

for index, row in sample_data.iterrows():
    for i in range(2, len(row), 3):
        if row[i] == 1:
            all_training_x_max = max(all_training_x_max, row[i+1])
            all_training_x_min = min(all_training_x_min, row[i+1])

            all_training_y_max = max(all_training_y_max, row[i+2])
            all_training_y_min = min(all_training_y_min, row[i+2])

print('all_training_x_max:', all_training_x_max)
print('all_training_x_min:', all_training_x_min)
print('all_training_y_max:', all_training_y_max)
print('all_training_y_min:', all_training_y_min)

dataset_split_list = ['TrainingData', 'ValidData', 'TestData']

for space in dataset_split_list:
    if space == 'TrainingData':
        csv_file = './data/Github_csv/TrainingData.csv'
        root = './data/Github_csv/' + 'TrainingData_roi.csv'
    elif space == 'ValidData':
        csv_file = './data/Github_csv/ValidData.csv'
        root = './data/Github_csv/' + 'ValidData_roi.csv'
    else:
        csv_file = './data/Github_csv/TestData.csv'
        root = './data/Github_csv/' + 'TestData_roi.csv'
        
    landmarks_frame = pd.read_csv(csv_file)
    length_csv = landmarks_frame.shape[0]

    new_title = ['img_path', 'scale', 'center_w', 'center_h', 'pixels', 'SELLA_v', 'SELLA_x', 'SELLA_y', 'NASION_v',
       'NASION_x', 'NASION_y', 'PORION_v', 'PORION_x', 'PORION_y',
       'ORBITALE_v', 'ORBITALE_x', 'ORBITALE_y', 'U I APEX_v', 'U I APEX_x',
       'U I APEX_y', 'POINT A_v', 'POINT A_x', 'POINT A_y', 'U I EDGE_v',
       'U I EDGE_x', 'U I EDGE_y', 'L I EDGE_v', 'L I EDGE_x', 'L I EDGE_y',
       'POINT B_v', 'POINT B_x', 'POINT B_y', 'L I APEX_v', 'L I APEX_x',
       'L I APEX_y', 'POGONION_v', 'POGONION_x', 'POGONION_y', 'MENTON_v',
       'MENTON_x', 'MENTON_y', 'U 6 APEX_v', 'U 6 APEX_x', 'U 6 APEX_y',
       'U 6 CUSP_v', 'U 6 CUSP_x', 'U 6 CUSP_y', 'L 6 CUSP_v', 'L 6 CUSP_x',
       'L 6 CUSP_y', 'L 6 APEX_v', 'L 6 APEX_x', 'L 6 APEX_y', 'GONION L_v',
       'GONION L_x', 'GONION L_y', 'GONION U_v', 'GONION U_x', 'GONION U_y',
       'CONDYLE_v', 'CONDYLE_x', 'CONDYLE_y', 'PNS_v', 'PNS_x', 'PNS_y',
       'BASION_v', 'BASION_x', 'BASION_y', 'U_6_MCP_v', 'U_6_MCP_x',
       'U_6_MCP_y', 'L_6_MCP_v', 'L_6_MCP_x', 'L_6_MCP_y', 'ANS_v', 'ANS_x',
       'ANS_y', 'ARTICULAR_v', 'ARTICULAR_x', 'ARTICULAR_y', 'ANAT_POR_v',
       'ANAT_POR_x', 'ANAT_POR_y']
    
    with open(root, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(new_title)
   
        for idx in range(length_csv):
            image_path = landmarks_frame.iloc[idx, 0]
            save_img_name = image_path[1:].split('/')[-1].split('.')[0]
            print(image_path)
            pixels = landmarks_frame.iloc[idx, 1]

            img = cv2.imread('./data/' + image_path[1:])  # np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
            print(img.shape)

            pts = landmarks_frame.iloc[idx, 2:].values
            pts = pts.astype('float').reshape(-1, 3)  # TrainingData (38,)

            v = pts[:, 0]
            x = pts[:, 1]
            x_copy = pts[:, 1].copy()
            y = pts[:, 2]
            y_copy = pts[:, 2].copy()
            
            width_max = img.shape[1]  # 
            height_max = img.shape[0] # 

            bounder = 10
            x_max = int(all_training_x_max) + bounder if int(all_training_x_max) + bounder <= width_max else width_max
            x_min = int(all_training_x_min) - bounder if int(all_training_x_min) - bounder >= 0 else 0
            y_max = int(all_training_y_max) + bounder if int(all_training_y_max) + bounder <= height_max else height_max
            y_min = int(all_training_y_min) - bounder if int(all_training_y_min) - bounder >= 0 else 0
            
            print(x_min, y_min, x_max, y_max)
            center_w = (x_min + x_max) / 2.0
            center_h = (y_min + y_max) / 2.0
            scale = max((x_max -x_min), (y_max - y_min)) / 200.0
            add_list =[image_path, scale, center_w, center_h, pixels]
            
            for ids in range(26):
                add_list.append(int(v[ids]))
                add_list.append(x_copy[ids])
                add_list.append(y_copy[ids])
            writer.writerow(np.array(add_list))
