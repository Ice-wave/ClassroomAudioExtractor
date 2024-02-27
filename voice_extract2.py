import json
import librosa
import numpy as np
from sklearn.cluster import KMeans
import soundfile as sf
import os

def load_timestamps(text_file_path):
    with open(text_file_path, 'r', encoding='utf-8-sig') as file:
        data = json.load(file)
    return [(entry['bg'], entry['ed']) for entry in data]

def extract_features(audio_file_path, timestamps, sr=None):
    audio_data, sr = librosa.load(audio_file_path, sr=sr)
    features = []
    for start_ms, end_ms in timestamps:
        start_sample, end_sample = int(start_ms / 1000.0 * sr), int(end_ms / 1000.0 * sr)
        clip = audio_data[start_sample:end_sample]
        mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=20)
        features.append(np.mean(mfcc, axis=1))
    return np.array(features)

def find_most_common_cluster(features):
    kmeans = KMeans(n_clusters=12, random_state=0).fit(features)
    cluster_counts = np.bincount(kmeans.labels_)
    most_common_cluster = np.argmax(cluster_counts)
    return most_common_cluster, kmeans

def save_cluster_center_clip(audio_file_path, timestamps, cluster_indices, kmeans, most_common_cluster, base_name, folder_path):
    distances = np.linalg.norm(cluster_indices - kmeans.cluster_centers_[most_common_cluster], axis=1)
    nearest_index = np.argmin(distances)
    start_ms, end_ms = timestamps[nearest_index]
    audio_data, sr = librosa.load(audio_file_path, sr=None)
    start_sample, end_sample = int(start_ms / 1000.0 * sr), int(end_ms / 1000.0 * sr)
    clip = audio_data[start_sample:end_sample]
    clip_filename = os.path.join(folder_path, f'{base_name}_most_common_cluster_center_clip.wav')
    sf.write(clip_filename, clip, sr)
    
    return clip_filename



# 文件夹路径定义
folder_path = 'voice/'

# 遍历voice文件夹
for file in os.listdir(folder_path):
    if file.endswith('.txt'):
        base_name = file[:-4]
        text_file_path = os.path.join(folder_path, file)
        audio_file_path = os.path.join(folder_path, base_name + '.wav')

        if not os.path.exists(audio_file_path):
            continue

        timestamps = load_timestamps(text_file_path)
        features = extract_features(audio_file_path, timestamps)
        most_common_cluster, kmeans = find_most_common_cluster(features)
        cluster_indices = np.where(kmeans.labels_ == most_common_cluster)[0]
        cluster_features = features[cluster_indices]
        clip_filename = save_cluster_center_clip(audio_file_path, timestamps, cluster_features, kmeans, most_common_cluster, base_name, folder_path)
        print(f"文件 {base_name} 的出现频率最多的聚类{most_common_cluster}的中心音频段已保存为 {clip_filename}。")
