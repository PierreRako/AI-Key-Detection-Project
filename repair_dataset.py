import glob
import os
import ntpath

data_to_repair = "/home/mirado/GIT/AI-Key-Detection-Project/Datasets/giantsteps-mtg-key-dataset-master/annotations/key"

keys_lower_diese = 'a,a#,b,c,c#,d,d#,e,f,f#,g,g#'
keys_lower_bemol = 'a,ab,b,bb,c,d,db,e,eb,f,g,gb'
keys_upper_diese = keys_lower_diese.upper()
Keys = keys_lower_diese.split(",") + keys_upper_diese.split(",") + keys_lower_bemol.split(",")
Tonality = ["major", "minor", "m", "M", "Major"]

def repair_dataset(key_annotation_path):
    key_annotations = glob.glob(os.path.join(key_annotation_path, '*'))
    total_number_files = len(key_annotations)
    counter=0

    newDirName= str(input("Enter the new name of the fodler: "))

    while(os.path.exists(newDirName)):
        newDirName = str(input("Input non existing fodler: "))

    separator = str(input("Enter separator, if its a space enter 'space':"))
    print("you selected: " + separator)

    if (separator == "space"):
        separator = " "

    os.makedirs(newDirName)

    for keyFile in key_annotations:
        split_ok = False
        annotation_ok = False

        f = open(keyFile, "r")
        line = f.readline()
        f.close()
        
        line = line.split(separator)
        if(len(line) >= 2):
            split_ok = True
        
        if split_ok == True:

            annotation = [line[0], line[1][:5]]
            #print(annotation)

            if (annotation[0] in Keys and annotation[1] in Tonality):
                annotation_ok = True

            if annotation_ok:
                curedAnnotation = cure_annotation(annotation)
                fileName = path_leaf(keyFile)
                fileName = newDirName + "/" + fileName
                f = open(fileName, "w")
                f.write(curedAnnotation)
                f.close()
                counter+=1
            else:
                print("Weird annotation")
        else:
            print("split failed because of weird annotation")

    print("processed "+ str(counter)+" files over "+str(total_number_files)+" files")
    return True

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def cure_annotation(annotation):
    annotation[0] = annotation[0].upper()

    if (annotation[1]=="M"):
        annotation[1] = "major"
    elif (annotation[1] == "m"):
        annotation[1] = "minor"

    annotation[1] = annotation[1].lower()

    if (len(annotation[0])>1):
        annotation[0] = handleAlteration(annotation[0])

    newAnnotation = annotation[0] + " " + annotation[1]
    return newAnnotation

def handleAlteration(entry):
    if (entry == "G#" or entry=="AB"):
        return "Ab"
    elif (entry == "A#" or entry=="BB"):
        return "Bb"
    elif (entry == "F#" or entry=="GB"):
        return "Gb"
    elif (entry == "C#" or entry=="DB"):
        return "Db"
    elif (entry=="D#" or entry=="EB"):
        return "Eb"
    else:
        return entry

repair_dataset(data_to_repair)