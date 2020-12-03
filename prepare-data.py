import numpy as np
import librosa, librosa.display;
import matplotlib.pyplot as plt
print("hello world")

FIG_SIZE = (15,10)

file = "Chill.mp3"

# load audio file with Librosa
signal, sample_rate = librosa.load(file, sr=44100)
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

#%% FUNCTION TO TRANSFORM A WAVEFORM INTO A CHROMAGRAM OF LENGTH 60 FRAMES

# output of this function is a 12*60 matrix which represents the chromgram 
# of (randomly chosen) 11sc of an input song. This can be used to create a dataset 
# that can be directly fed to any neural network.

def our_chromagram(signal):
    l = len(signal)
    offset = np.random.randint(l - 70*hopLength)
    croppedSig = signal[offset: offset + 59*frameSize + 1]
    #cropSigDur = len(croppedSig)/sr
    #print("New signal duration : ", cropSigDur)
    return librosa.feature.chroma_stft(croppedSig, sr, n_fft= frameSize, hop_length = hopLength)

#%% Test Function our_chromagram
chroma = our_chromagram(signal)
print("dimension chromagramme : ", chroma.shape)
librosa.display.specshow(chroma)
plt.show()

#%% FUNCTION TO TRANSFORM A WAVEFORM INTO A SPECTROGRAM WHICH IS LINEAR WITH RESPECT TO SEMITONES

## 
def our_spectrogram(signal):
    return None