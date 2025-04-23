import os  
import cv2  
import pandas as pd  
import numpy as np  
from scipy import stats  

# Root folder dataset  
root_dir = 'dataset-ayam-sapi-final'  
subfolders = ['train', 'test', 'valid']  
folders = [os.path.join(root_dir, sf) for sf in subfolders]  

def compute_stat_features(channel_data):  
    if len(channel_data) < 2:  
        diff = np.array([0])  
    else:  
        diff = np.diff(channel_data)  

    return {  
        'mean': np.mean(diff),  
        'std': np.std(diff),  
        'skewness': stats.skew(diff),  
        'kurtosis': stats.kurtosis(diff),  
        'median': np.median(diff),  
        'min': np.min(diff),  
        'max': np.max(diff)  
    }  

features_list = []  

for folder in folders:  
    # Dapatkan nama file gambar di folder  
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]  

    for filename in image_files:  
        img_path = os.path.join(folder, filename)  
        image = cv2.imread(img_path)  
        if image is None:  
            print(f"Warning: gagal baca gambar {img_path}")  
            continue  
        
        # Jika kamu punya file anotasi untuk kelas, kamu bisa baca kelasnya juga  
        # tapi di sini saya asumsikan kelas tidak ada atau bisa diperoleh dengan cara lain  
        label = None  # atau isi sesuai kebutuhan  

        # Pisahkan channel: BGR ke RGB  
        R = image[:, :, 2].flatten()  
        G = image[:, :, 1].flatten()  
        B = image[:, :, 0].flatten()  

        # Hitung fitur orde1 per channel  
        R_feats = compute_stat_features(R)  
        G_feats = compute_stat_features(G)  
        B_feats = compute_stat_features(B)  

        feat = {'filename': filename}  

        # gabungkan fitur ke dict  
        for stat_name, val in R_feats.items():  
            feat[f'R_{stat_name}'] = val  
        for stat_name, val in G_feats.items():  
            feat[f'G_{stat_name}'] = val  
        for stat_name, val in B_feats.items():  
            feat[f'B_{stat_name}'] = val  

        if label is not None:  
            feat['class'] = label  

        features_list.append(feat)  

df_features = pd.DataFrame(features_list)  
output_file = 'Orde1_Daging.csv'  
df_features.to_csv(output_file, index=False)  
print(f"Fitur orde 1 per gambar disimpan di {output_file}")  