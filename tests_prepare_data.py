#%% imports

import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from prepare_data import prepare_data1
from prepare_data import prepare_data2

FIG_SIZE = (15,10)

file = "Chill.mp3"

# load audio file with Librosa
signal, sample_rate = librosa.load(file, sr=44100)

#%% Data preparation constants

nbOfFrames = 60
sr = 44100
frameSize = 8192
hopLength = frameSize

fMin=librosa.note_to_hz('C2')
nBins=120
binsPerOctave = 24


#%% Test Function prepare_data1
def test_prepare_1():
    chroma = prepare_data1(signal)
    print("dimension chromagramme : ", chroma.shape)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(chroma,hop_length=hopLength,
                                sr=44100, x_axis='s', y_axis='chroma', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='Chromagram')
    plt.show()

#%% test function prepare_data2
def test_prepare_2():
    C = prepare_data2(signal)
    print("dimension spectrogramme : ", C.shape)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),hop_length=hopLength,
    bins_per_octave=binsPerOctave, fmin=fMin,sr=sr, x_axis='s', y_axis='cqt_note', ax=ax)
    ax.set_title('Constant-Q power spectrum')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.show()

#%% actual tests

test_prepare_2()