import os  
import cv2  
import pandas as pd  
import numpy as np  

# Folder yang akan diproses  
folders = ['train', 'test', 'valid']  

def process_folder(folder):  
    annotation_path = os.path.join(folder, '_annotations.csv')  
    df_annot = pd.read_csv(annotation_path)  
    
    features_list = []  
    
    for idx, row in df_annot.iterrows():  
        img_path = os.path.join(folder, row['filename'])  
        image = cv2.imread(img_path)  
        if image is None:  
            print(f"Warning: gagal baca gambar {img_path}")  
            continue  
        
        # bounding box koordinat  
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])  
        
        # pastikan koordinat valid dan dalam batas gambar  
        xmin = max(0, xmin)  
        ymin = max(0, ymin)  
        xmax = min(image.shape[1]-1, xmax)  
        ymax = min(image.shape[0]-1, ymax)  
        
        if xmax <= xmin or ymax <= ymin:  
            print(f"Warning: bounding box tidak valid di {img_path}")  
            continue  
        
        roi = image[ymin:ymax, xmin:xmax]  
        
        # Hitung rata-rata RGB (cv2 baca warna BGR, kita ubah ke RGB)  
        mean_bgr = cv2.mean(roi)[:3]  # ambil 3 channel  
        mean_rgb = mean_bgr[::-1]     # balik ke RGB  
        
        # Simpan fitur dan label  
        features = {  
            'filename': row['filename'],  
            'R_mean': mean_rgb[0],  
            'G_mean': mean_rgb[1],  
            'B_mean': mean_rgb[2],  
            'class': row['class']  
        }  
        features_list.append(features)  
    
    return pd.DataFrame(features_list)  

# Proses ketiga folder dan gabungkan hasilnya  
df_all = pd.DataFrame()  
for folder in folders:  
    df_folder = process_folder(folder)  
    df_all = pd.concat([df_all, df_folder], ignore_index=True)  

# Simpan ke file CSV untuk WEKA  
output_csv = 'Data_RGB_Protein_Base_On_Bonding_Box.csv'  
df_all.to_csv(output_csv, index=False)  

print(f"Proses selesai, data disimpan di {output_csv}")  