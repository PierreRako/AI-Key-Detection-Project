import matplotlib.pyplot as plt
import os
import glob
from prepare_data import prepare_data1 
import librosa, librosa.display
from PIL import Image
import numpy as np

#%% Converting data files into PNG images = Extracting spectrogram
cmap = plt.get_cmap('inferno')
keys = 'A major,A minor,Ab major,Ab minor,B major,B minor,Bb major,Bb minor,C major,C minor,D major,D minor,Db major,Db minor,E minor,E major,Eb major,Eb minor,F minor,F major,G major,G minor,Gb major,Gb minor'.split(",")

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

def convert_audio_to_chroma_images(data_set_path, key_annotation_path):
    audio_key_tuples = load_dataset(data_set_path, key_annotation_path)

    j = 1
    for i in range(len(audio_key_tuples)):
        #Extracting the fileName from the path
        fileName = fileName_from_path(audio_key_tuples[i][0])
        
        #if (j==2):
        #    break

        signal, sample_rate = librosa.load(audio_key_tuples[i][0], sr=44100)
        chroma = prepare_data1(signal)
        plt.figure()
        librosa.display.specshow(chroma)
        plt.axis('off')
        plt.savefig(f'img_data/{fileName}.png')
        plt.close()

        #Setting the path to the image
        #cwd = os.getcwd()
        #audio_key_tuples[i][0] = glob.glob(os.path.abspath(f'{cwd}/img_data/{fileName}.png'))
        audio_key_tuples[i][0] = os.path.relpath(f'img_data/{fileName}.png')

        #Adding the name of the file to the list
        audio_key_tuples[i] = [str(fileName)] + audio_key_tuples[i]

        print(f"Image n°{j} processed: " + audio_key_tuples[i][0] + " " + str(audio_key_tuples[i][1]) + " " + str(audio_key_tuples[i][2]))

        j+=1

    return audio_key_tuples

# A function to prepare the panda dataframe
# @param 
# @return

def prepare_panda_dataFrame(data_set_path, key_annotation_path):
    audio_key_tuples = convert_audio_to_chroma_images(data_set_path, key_annotation_path)

    #Converting all images to numpy vectors
    for i in range(len(audio_key_tuples)):
        audio_key_tuples[i][1] = create_numpy_1D_array_from_image(audio_key_tuples[i][1])
        audio_key_tuples[i].append(keys.index(audio_key_tuples[i][2]))

    #Converting the list into a numpy array
    audio_key_tuples = np.array(audio_key_tuples)

    namesList = audio_key_tuples[:,0]
    chroma_vectorList = audio_key_tuples[:, 1]
    keyList = audio_key_tuples[:, 2]
    key_codeList = audio_key_tuples[:, 3]

    return audio_key_tuples

def create_numpy_1D_array_from_image(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    img_vector = np.reshape(img_array, -1)

    return img_vector