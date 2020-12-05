#%% imports

import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt

#%% Data preparation constants
nbOfFrames = 60
sr = 44100
frameSize = 8192
hopLength = frameSize

#%% Print data information

hopLengthDuration = float(hopLength)/sr
frameDuration = float(frameSize)/sr
#print("STFT hop length duration is: {}sc".format(hopLengthDuration))
#print("STFT window duration is: {}sc".format(frameDuration))

#%% FUNCTION TO TRANSFORM A WAVEFORM INTO A CHROMAGRAM OF LENGTH 60 FRAMES

# output of this function is a 12*60 matrix which represents the chromgram 
# of (randomly chosen) 11sc of an input song. This can be used to create a dataset 
# that can be directly fed to any neural network.

def prepare_data1(signal):
    l = len(signal)
    offset = np.random.randint(l - 70*hopLength)
    croppedSig = signal[offset: offset + 59*frameSize + 1]
    # cropSigDur = len(croppedSig)/sr
    # print("New signal duration : ", cropSigDur)
    return librosa.feature.chroma_stft(croppedSig, sr, n_fft= frameSize, hop_length = hopLength)


#%% Constants for 2nd function

fMin=librosa.note_to_hz('C2')
nBins=120
binsPerOctave = 24

#%% FUNCTION TO TRANSFORM A WAVEFORM INTO A SPECTROGRAM WHICH IS LINEAR WITH RESPECT TO SEMITONES

# output of this function is a 120*60 matrix which represents the constantQ 
# tranform of (randomly chosen) 11sc of the input array signal. The constantQ 
# is a special type of spectrogram which re-scale the frequency-axis to match
# the well-tempered scale of music : we uses a scale of 24 frequencies/octave 
# and take into account 5 octave starting from C2 (65.41Hz)

def prepare_data2(signal):
    l = len(signal)
    offset = np.random.randint(l - 70*hopLength)
    croppedSig = signal[offset: offset + 59*frameSize + 1]
    return np.abs(
        librosa.cqt(
            croppedSig,sr,hop_length=hopLength, fmin=fMin,
            n_bins=nBins,bins_per_octave=binsPerOctave
        )
    )

