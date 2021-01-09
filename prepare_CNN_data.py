#import matplotlib.pyplot as plt
import os
import glob
from prepare_data import prepare_data2
import librosa, librosa.display
#from PIL import Image
import numpy as np
#import pandas as pd

#%% Converting data files into PNG images = Extracting spectrogram
#cmap = plt.get_cmap('inferno')
keys = 'A major,A minor,Ab major,Ab minor,B major,B minor,Bb major,Bb minor,C major,C minor,D major,D minor,Db major,Db minor,E minor,E major,Eb major,Eb minor,F minor,F major,G major,G minor,Gb major,Gb minor'.split(",")

# @return fileName without any extension
def fileName_from_path(path):
    fileNameFull = path.split("/")[-1]
    splitName = fileNameFull.split(".")[:-1]
    
    separator = "."
    fileName = separator.join(splitName)
    return fileName

#%%---------------------------------------------------------------------------------------------------------------------------

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

#%%---------------------------------------------------------------------------------------------------------------------------

# A function to load a dataset (folder containing all audios)
# It associates the path of an audio to its corresponding key
# @param data_set_path : path to the folder containing all audio
# @param key_annotation_path : path to the folder containing all key annotations
# @return list of tuples [audio, key]
def load_dataset(data_set_path, key_annotation_path):
    print(f"loading {data_set_path}")
    #Getting all audio files
    files = glob.glob(os.path.join(data_set_path, '*.wav'))

    #Getting key_annotations
    key_annotations = glob.glob(os.path.join(key_annotation_path, '*'))

    audio_key_tuples = zip_audio_key(files, key_annotations)

    return audio_key_tuples

#%%---------------------------------------------------------------------------------------------------------------------------

# A function to convert audio files of a dataset to chroma image and save it
# @param data_set_path

def convert_audio_to_chroma_or_spectro(data_set_path, key_annotation_path):
    audio_key_tuples = load_dataset(data_set_path, key_annotation_path)

    for i in range(len(audio_key_tuples)):
        #Extracting the fileName from the path
        fileName = fileName_from_path(audio_key_tuples[i][0])

        signal, sample_rate = librosa.load(audio_key_tuples[i][0], sr=44100)
        
        transfo_matrix = prepare_data2(signal)

        # Saving chroma to an image file
        #plt.figure()
        #librosa.display.specshow(transfo_matrix)
        #plt.axis('off')
        #plt.savefig(f'img_data/{fileName}.jpg')
        #plt.close()

        #Setting the path to the image
        audio_key_tuples[i][0] = transfo_matrix

        #Adding the name of the file to the list
        audio_key_tuples[i] = [str(fileName)] + audio_key_tuples[i]

    print("Audios have correctly been processed")

    return audio_key_tuples

#%%---------------------------------------------------------------------------------------------------------------------------    

# A function to prepare the panda dataframe from the audios of a dataset and its key annotations
# @param data_set_path : path to the folder containing all audio
# @param key_annotation_path : path to the folder containing all key annotations
# @return df : panda dataframe of the dataset 
# 4 columns: filename, chromagram, key, codedkey

def prepare_cnn_data(data_set_path, key_annotation_path):
    audio_key_tuples = convert_audio_to_chroma_or_spectro(data_set_path, key_annotation_path)

    #Adding coded keys to audio_key_tuples
    for i in range(len(audio_key_tuples)):

        audio_key_tuples[i].append(keys.index(audio_key_tuples[i][2]))
    
    #Converting the list into a numpy array
    audio_key_tuples = np.array(audio_key_tuples)
    #print(audio_key_tuples.shape)
    namesList = audio_key_tuples[:,0]
    chroma_matricesList = audio_key_tuples[:, 1]
    keyList = audio_key_tuples[:, 2]
    key_codeList = audio_key_tuples[:, 3]

    '''
    # Creating the dataframe using a dictionnary
    dictionnary = {'filename':namesList, 'chromagram':chroma_matricesList, 'key':keyList, 'coded_key':key_codeList}
    df = pd.DataFrame(data=dictionnary)

    # Letting the user choose the name of the csv file
    csv_file_name = str(input("Enter the name of your dataset:"))
    csv_file_name += ".csv"
    file_already_exists = glob.glob(csv_file_name)

    while file_already_exists:
        csv_file_name = str(input("This dataset already has a csv file, enter another name:"))
        csv_file_name += ".csv"
        file_already_exists = glob.glob(csv_file_name)

    # Saving the dataframe to a csv file
    df.to_csv(csv_file_name, index=False, encoding='utf8')
    '''
    return (namesList,np.stack(chroma_matricesList),keyList,key_codeList.astype('int'))


#%%---------------------------------------------------------------------------------------------------------------------------
''' Useless
def create_numpy_1D_array_from_image(image_path, bool):
    img = Image.open(image_path)
    img_array = np.asarray(img)

    if bool:
        print(img_array)

    img_vector = np.reshape(img_array, -1)

    return img_vector
'''
#---------------------------------------------------------------------------------------------------------------------------