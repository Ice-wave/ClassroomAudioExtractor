import json
import librosa
import soundfile as sf

text_file_path = r'voice/text1.txt'  # Use forward slash for cross-platform compatibility
audio_file_path = 'voice/text1.wav'
# Initialize a list to hold the timestamps and possibly transcribed text
timestamps = []

# Read and parse the text file
with open(text_file_path, 'r', encoding='utf-8-sig') as file:
    data = json.load(file)  # Assuming the file is in JSON format
    
    # Loop through each entry to extract timestamps and 'onebest' if available
    for entry in data:
        bg = entry['bg']  # Beginning timestamp
        ed = entry['ed']  # End timestamp
        timestamps.append((bg, ed))

for ts in timestamps[:5]:
    print(ts)



# 加载音频文件
audio_data, sr = librosa.load(audio_file_path, sr=None)

# 对每个时间戳进行循环处理
for i, (start_ms, end_ms) in enumerate(timestamps):
    if i>4:
        break
    # 将开始和结束时间从毫秒转换为秒
    start_sec = start_ms / 1000.0
    end_sec = end_ms / 1000.0

    # 计算开始和结束的采样点
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    
    # 截取音频片段
    clip = audio_data[start_sample:end_sample]
    
    # 为每个片段设置文件名并保存
    clip_path = f'voice/clip_{i}.wav'
    sf.write(clip_path, clip, sr)


