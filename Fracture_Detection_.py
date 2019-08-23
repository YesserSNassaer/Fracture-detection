# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 12:04:08 2019
@author: Yesser H. Nasser
"""
import numpy as np 
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras import optimizers
from keras.utils import np_utils
import keras

from PIL import Image
import requests
from io import BytesIO
import os
import pickle
import tqdm
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels

df = pd.read_csv('Fractures_data.csv')
#df['label'] = df.annotation.apply(lambda x: x['labels'][0] if len(x['labels'])==1 else 'Fracture')
print(df.shape)
df.head()

images = []

for link in tqdm.tqdm(df['content']):
    response = requests.get(link)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))
    numpy_img = img_to_array(img)
    img_batch = np.expand_dims(numpy_img, axis=0)
    images.append(img_batch.astype('float16'))
    
images = np.vstack(images)
print(images.shape)

# plot random images from the dataset
random_id = np.random.randint(0,images.shape[0],25)
f, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize = (15,15))

for ax, img, title in zip(axes.ravel(), images[random_id], df['label'][random_id]):
    ax.imshow(array_to_img(img))
    ax.set_title(title)
    plt.tight_layout()

''' ========================================================================'''
''' ========== To learn more about the code please get in touch ============'''
'''==================== yesser.nasser@icloud.com ==========================='''
''' ========================================================================'''
