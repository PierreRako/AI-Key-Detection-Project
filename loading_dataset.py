import matplotlib.pyplot as plt
import os
import glob
from prepare_data import prepare_data1 
import librosa, librosa.display

#%% Converting data files into PNG images = Extracting spectrogram
cmap = plt.get_cmap('inferno')
keys = 'A major, A minor, Ab major, Ab minor, B major, B minor, Bb minor, Bb major, C major, C minor, D major, D minor, Db minor, Db major, E minor, E major, Eb minor, Eb major, F minor, F major, G major, G minor, Gb minor, Gb major'.split(",")

# @return fileName without any extension
def fileName_from_path(path):
    fileNameFull = path.split("/")[-1]
    splitName = fileNameFull.split(".")[:-1]
    
    separator = "."
    fileName = separator.join(splitName)
    return fileName

# @param files is a list of paths to files
# @param keys is a list of pahts to key files
def zip_audio_key(files, keys):
    orderedKeyFiles = []

    keyNames = [fileName_from_path(key) for key in keys]

    for file in files:
        fileName = fileName_from_path(file)

        index = keyNames.index(fileName)

        keyFile = open(keys[index], "r")
        key = keyFile.read()

        tuple = [file, key]
        orderedKeyFiles.append(tuple)

    return orderedKeyFiles

# A function to load a dataset (folder containing all audios)
# It associates the path of an audio to its corresponding key
# @param data_set_path User has to specify the paf to the folder containing all audio
# @param key_annotation_path
# @return list of tuples [audio, key]
def load_dataset(data_set_path, key_annotation_path):
    print(f"loading {data_set_path}")
    #Getting all audio files
    files = glob.glob(os.path.join(data_set_path, '*.wav'))

    #Getting key_annotations
    key_annotations = glob.glob(os.path.join(key_annotation_path, '*'))

    audio_key_tuples = zip_audio_key(files, key_annotations)

    return audio_key_tuples

# A function to convert audio files of a dataset to chroma image and save it
# @param data_set_path

def convert_to_chroma_images(data_set_path, key_annotation_path):
    audio_key_tuples = load_dataset(data_set_path, key_annotation_path)

    i = 0
    for tuple in audio_key_tuples:
        #Extracting the fileName from the path
        fileName = fileName_from_path(tuple[0])
        
        if (i!=0):
            break
        
        signal, sample_rate = librosa.load(tuple[0], sr=44100)
        chroma = prepare_data1(signal)
        plt.figure()
        librosa.display.specshow(chroma)
        plt.axis('off')
        plt.savefig(f'img_data/{fileName}.png')
        plt.clf()

        #Setting the path to the image
        cwd = os.getcwd()
        tuple[0] = glob.glob(os.path.abspath(f'{cwd}/img_data/{fileName}.png'))

        print(tuple[0])
        print(tuple[1])
        i+=1
    return 0