import matplotlib.pyplot as plt
import os
import glob

#%% Converting data files into PNG images = Extracting spectrogram
cmap = plt.get_cmap('inferno')
keys = 'A major, A minor, Ab major, Ab minor, B major, B minor, Bb minor, Bb major, C major, C minor, D major, D minor, Db minor, Db major, E minor, E major, Eb minor, Eb major, F minor, F major, G major, G minor, Gb minor, Gb major'.split(",")

# A function to load a dataset (folder containing all audios)
# @param data_set_path User has to specify the paf to the folder containing all audio
# @return paths of each file in the dataset
def load_dataset(data_set_path):
    print(f"loading {data_set_path}")
    files = glob.glob(os.path.join(data_set_path, '*.wav'))
    return files

# A function to convert audio files of a dataset to chroma image and save it
# @param data_set_path
# @return 

def convert_to_chroma_images(data_set_path):
    files = load_dataset(data_set_path)
    

load_dataset("/home/mirado/GIT/AI-Key-Detection-Project/Datasets/giantsteps-key-dataset-master/audio")
