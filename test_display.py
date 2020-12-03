
import numpy as np
import librosa, librosa.display;
import matplotlib.pyplot as plt
print("hello world")

FIG_SIZE = (15,10)

file = "Chill.mp3"

# load audio file with Librosa
signal, sample_rate = librosa.load(file, sr=44100)
print("longueur du signal : ", signal.shape)
#%% Data preparation constants
nbOfFrames = 60
sr = 44100
binsPerOctave = 24
frameSize = 8192
hopLength = frameSize

#%% Print data information

hopLengthDuration = float(hopLength)/sr
frameDuration = float(frameSize)/sr
print("STFT hop length duration is: {}sc".format(hopLengthDuration))
print("STFT window duration is: {}sc".format(frameDuration))

#%% Various Tests
'''
chroma = librosa.feature.chroma_stft(signal, sr, n_fft= frameSize, hop_length = hopLength)
print("dimension chromagramme : ", chroma.shape)
fig, ax = plt.subplots()
img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='s', ax=ax)
fig.colorbar(img, ax=ax)
ax.set(title='Chromagram')
plt.show()
'''

C = np.abs(librosa.cqt(signal,sr,hop_length=hopLength, fmin=librosa.note_to_hz('C2'),
n_bins=120,bins_per_octave=binsPerOctave))
print("dimension spectrogramme : ", C.shape)
fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),hop_length=hopLength, bins_per_octave=24,
                               sr=44100, x_axis='s', y_axis='cqt_note', ax=ax)
ax.set_title('Constant-Q power spectrum')
fig.colorbar(img, ax=ax, format="%+2.0f dB")
plt.show()