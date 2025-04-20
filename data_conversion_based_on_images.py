# Conversion Base On Numbers Images

import os  
import cv2  
import pandas as pd  
import numpy as np  

folders = ['train', 'test', 'valid']  

def process_folder(folder):  
    annotation_path = os.path.join(folder, '_annotations.csv')  
    df_annot = pd.read_csv(annotation_path)  
    
    data_by_image = {}  
    
    for idx, row in df_annot.iterrows():  
        img_path = os.path.join(folder, row['filename'])  
        image = cv2.imread(img_path)  
        if image is None:  
            print(f"Warning: gagal baca gambar {img_path}")  
            continue  
        
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])  
        xmin = max(0, xmin)  
        ymin = max(0, ymin)  
        xmax = min(image.shape[1]-1, xmax)  
        ymax = min(image.shape[0]-1, ymax)  
        
        if xmax <= xmin or ymax <= ymin:  
            print(f"Warning: bounding box tidak valid di {img_path}")  
            continue  
        
        roi = image[ymin:ymax, xmin:xmax]  
        mean_bgr = cv2.mean(roi)[:3]  
        mean_rgb = mean_bgr[::-1]  
        
        filename = row['filename']  
        label = row['class']  
        
        if filename not in data_by_image:  
            data_by_image[filename] = {  
                'R': [],  
                'G': [],  
                'B': [],  
                'class': label  
            }  
        
        data_by_image[filename]['R'].append(mean_rgb[0])  
        data_by_image[filename]['G'].append(mean_rgb[1])  
        data_by_image[filename]['B'].append(mean_rgb[2])  
    
    # Sekarang rata-rata semua bbox per gambar  
    features_list = []  
    for filename, vals in data_by_image.items():  
        R_mean = np.mean(vals['R'])  
        G_mean = np.mean(vals['G'])  
        B_mean = np.mean(vals['B'])  
        features_list.append({  
            'filename': filename,  
            'R_mean': R_mean,  
            'G_mean': G_mean,  
            'B_mean': B_mean,  
            'class': vals['class']  
        })  
    return pd.DataFrame(features_list)  

df_all = pd.DataFrame()  
for folder in folders:  
    df_folder = process_folder(folder)  
    df_all = pd.concat([df_all, df_folder], ignore_index=True)  

output_csv = 'Data_RGB_Protein_Base_On_Images.csv'  
df_all.to_csv(output_csv, index=False)  

print(f"Proses selesai, data disimpan di {output_csv}")  