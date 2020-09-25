# import packets
import numpy as np #to process image matrices
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt #display result
import os #access file system to read an image from train and test directory from our machines
import glob

# load dataset
f = open("miniCategories.txt", "r") #52 classes with animals/living things
# For reading use
classes = f.readlines()
f.close()
classes = [c.replace('\n', '').replace(' ', '_') for c in classes]

import urllib.request

#download classes from miniCategories.txt
def download():
    #quickdraw dataset
    base = 'https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap;tab=objects?pli=1&prefix=&forceOnObjectsSortingFiltering=false'
    for c in classes:
        cls_url = c.replace('_', '%20')
        path = base+cls_url+'.npy'
        #print(path)
        urllib.request.urlretrieve(path, 'data/'+c+'.npy')

#download() #only need to be called once

def load_data(path, ratio= 0.1, itemsPerClass = 2000):
    files = glob.glob(os.path.join(path, '*.npy'))

#
