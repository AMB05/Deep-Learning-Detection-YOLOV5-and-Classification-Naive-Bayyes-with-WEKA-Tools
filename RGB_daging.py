import os  
import cv2  
import pandas as pd  
import numpy as np  

# Path root dataset  
root_dir = 'dataset-ayam-sapi-final'  

# List folder lengkap dengan path  
folders = [os.path.join(root_dir, subfolder) for subfolder in ['train', 'test', 'valid']]  

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

        # Hitung rata-rata RGB seluruh gambar  
        mean_bgr = cv2.mean(image)[:3]  # OpenCV default BGR  
        mean_rgb = mean_bgr[::-1]       # Balik ke RGB  
        
        features = {  
            'filename': row['filename'],  
            'R_mean': mean_rgb[0],  
            'G_mean': mean_rgb[1],  
            'B_mean': mean_rgb[2],  
            'class': row['class']  
        }  
        features_list.append(features)  

    return pd.DataFrame(features_list)  

df_all = pd.DataFrame()  
for folder in folders:  
    df_folder = process_folder(folder)  
    df_all = pd.concat([df_all, df_folder], ignore_index=True)  

output_csv = 'RGB_Daging.csv'  
df_all.to_csv(output_csv, index=False)  

print(f"Proses selesai, data disimpan di {output_csv}")  