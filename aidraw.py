# import packets
import numpy as np #to process image matrices
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt #display result

# load dataset
f = open("miniCategories.txt", "r") #52 classes with animals/living things
# For reading use
classes = f.readinglines()
f.close()
classes = [c.replace('\n', '').replace(' ', '_') for c in classes]
