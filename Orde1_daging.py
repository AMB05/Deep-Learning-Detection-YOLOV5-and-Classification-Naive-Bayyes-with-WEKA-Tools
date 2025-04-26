import os  
import cv2  
import pandas as pd  
import numpy as np  
from scipy import stats  

# Root folder dataset  
root_dir = 'dataset_Sapi_Fresh_Unfresh'  
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
    # Baca anotasi kelas jika ada  
    annot_path = os.path.join(folder, '_annotations.csv')  
    if os.path.exists(annot_path):  
        df_annot = pd.read_csv(annot_path)  
        # Buat peta filename ke class  
        filename_to_class = dict(zip(df_annot['filename'], df_annot['class']))  
    else:  
        filename_to_class = {}  

    # Dapatkan nama file gambar di folder  
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]  

    for filename in image_files:  
        img_path = os.path.join(folder, filename)  
        image = cv2.imread(img_path)  
        if image is None:  
            print(f"Warning: gagal baca gambar {img_path}")  
            continue  
        
        R = image[:, :, 2].flatten()  
        G = image[:, :, 1].flatten()  
        B = image[:, :, 0].flatten()  

        R_feats = compute_stat_features(R)  
        G_feats = compute_stat_features(G)  
        B_feats = compute_stat_features(B)  

        feat = {'filename': filename}  

        for stat_name, val in R_feats.items():  
            feat[f'R_{stat_name}'] = val  
        for stat_name, val in G_feats.items():  
            feat[f'G_{stat_name}'] = val  
        for stat_name, val in B_feats.items():  
            feat[f'B_{stat_name}'] = val  

        # Masukkan class kalau ada  
        feat['class'] = filename_to_class.get(filename, 'Unknown')  

        features_list.append(feat)  

df_features = pd.DataFrame(features_list)  
output_file = 'Orde1_Esi.csv'  
df_features.to_csv(output_file, index=False)  
print(f"Fitur orde 1 per gambar disimpan di {output_file}")  


