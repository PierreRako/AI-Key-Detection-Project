import pandas as pd
import librosa 

#%% Loading an audio file
audio_data='./giantsteps-key-dataset-master/audio/10089.LOFI.wav'
x, sr = librosa.load(audio_data)
#librosa.load(audio_data, sr=44100) resampling at 44100Hz
#librosa.load(audio_path, sr=None)

#%% Visualizing audio
import matplotlib.pyplot as plt
import librosa.display
plt.figure(figsize=(14,5))
librosa.display.waveplot(x, sr=sr)

#%% Spectrogram
X = librosa.stft(x) #Short term Fourier transform
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))

#normal scale
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()

#logarithmic scale
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()

#%% Chromagram
#chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
#plt.figure(figsize=(15, 5))
#librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
