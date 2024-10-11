from msilib.schema import Directory
from telnetlib import theNULL
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import PIL
import PIL.Image as Image
import glob
import sys

# example of loading an image with the Keras API
from keras.preprocessing.image import load_img
from keras.preprocessing.image import save_img
from keras import layers

print(tf.__version__)


i = 1
IMG_SIZE = 100

resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(IMG_SIZE, IMG_SIZE),
  layers.Rescaling(1./255)
])

data_augmentation1 = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.5),
])

data_augmentation2 = tf.keras.Sequential([
  layers.RandomContrast(0.2),
  layers.RandomZoom(0.1),
  layers.RandomTranslation(0.2,0.2)
])

data_augmentation3 = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomZoom(0.1),
])

data_augmentation4 = tf.keras.Sequential([
  layers.RandomContrast(0.2),
  layers.RandomRotation(0.5),
  layers.RandomTranslation(0.2,0.2)
])

data_augmentation5 = tf.keras.Sequential([
  layers.RandomRotation(0.5),
  layers.RandomZoom(0.1),
])



for subs in os.scandir('data/Raw'): 
    j = 1
    dest = 'data/Augmented/' + subs.name
    if not os.path.exists(dest):
        os.makedirs(dest)

    for filename in glob.glob('data/Raw/'+ subs.name +'/*.jpg'): 
        im=load_img(filename)
        aug_im = resize_and_rescale(im)
        #aug_im = tf.image.rgb_to_grayscale(aug_im)
        d = dest+'/' +str(j) 
        save_img(d+".jpg",aug_im)
        RF_img1 = data_augmentation1(aug_im)
        save_img(d+"-1.jpg",RF_img1)
        RF_img2 = data_augmentation2(aug_im)
        save_img(d+"-2.jpg",RF_img2)
        RF_img1 = data_augmentation1(aug_im)
        save_img(d+"-3.jpg",RF_img1)
        RF_img2 = data_augmentation2(aug_im)
        save_img(d+"-4.jpg",RF_img2)
        RF_img2 = data_augmentation3(aug_im)
        save_img(d+"-5.jpg",RF_img2)
        RF_img2 = data_augmentation4(aug_im)
        save_img(d+"-6.jpg",RF_img2)
        RF_img2 = data_augmentation3(RF_img1)
        save_img(d+"-7.jpg",RF_img2)
        RF_img2 = data_augmentation4(RF_img1)
        save_img(d+"-8.jpg",RF_img2)
        RF_img2 = data_augmentation5(RF_img1)
        save_img(d+"-9.jpg",RF_img2)
        j = j + 1
    i = 1 + 1

print(sys.version)