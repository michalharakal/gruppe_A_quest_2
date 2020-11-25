# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 08:10:25 2020

@author: Wolfgang Reuter

TODO: Get rid of hard coded values in build_generator() 
      and build_discriminator()
      
TODO: Get rid of hard coded path in save_images()
"""
# =============================================================================
# Imports
# =============================================================================

import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

# Model functions from keras
from keras.layers import Input, Reshape, Dropout, Dense, Flatten, \
                         BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model

# =============================================================================
# Functions
# =============================================================================

def get_random_noise(rows, columns, random_noise_dimension):
    """

    Parameters
    ----------
    rows : Integer
        Pixel size of the height of a "noise image"
    columns : Integer
        Pixel size of the width of an "noise image"
    random_noise_dimension : Integer
        Number of channels of a "noise image"

    Returns
    -------
    2-dim numpy array
        Array of shape (rows * columns, random_noise_dimensions) with 
        normally distributed values with mean 0 and standard deviation 1

    """
    return np.random.normal(0, 1, (rows * columns, random_noise_dimension))

def get_training_data(datafolder, image_width, image_height, channels):
    """
    Loads the training data from a specified folder
    
    Parameters
    ----------
    datafolder : String
        Path to directory where the training images are stored
    image_width : Integer
        Pixel width of the images (Nr. of columns)
    image_height : Integer
        Pixel height of the images (Nr. of rows)
    channels : Integer
        Number of color channels

    Returns
    -------
    training_data : 4-dim numpy array
        The training data as numpy array of shaape:    
            (Nr. images, image_height, image_width, Nr. channels) 

    """
    print("Loading training data...")

    training_data = []
    #Finds all files in datafolder
    filenames = os.listdir(datafolder)
    for filename in tqdm(filenames):
        #Combines folder name and file name.
        path = os.path.join(datafolder,filename)
        #Opens an image as an Image object.
        image = Image.open(path)
        #Resizes to a desired size.
        image = image.resize((image_width,image_height),Image.ANTIALIAS)
        #Creates an array of pixel values from the image.
        pixel_array = np.asarray(image)
        
        # Clip alpha channel, if existant
        if pixel_array.shape[2] > channels: 
            pixel_array = pixel_array[:,:,:channels]

        training_data.append(pixel_array)

    #training_data is converted to a numpy array
    training_data = \
        np.reshape(training_data,(-1, image_width, image_height, channels))
    return training_data

def build_generator(random_noise_dimension, channels):
    #Generator attempts to fool discriminator by generating new images.
    model = Sequential()

    model.add(Dense(256*4*4,activation="relu",\
                    input_dim=random_noise_dimension))
    model.add(Reshape((4,4,256)))

    # Four layers of upsampling, convolution, batch normalization 
    # and activation.
    # 1. Upsampling: Input data is repeated. Default is (2,2). 
    #    In that case a 4x4x256 array becomes an 8x8x256 array.
    # 2. Convolution: If you are not familiar, you should watch 
    #    this video: https://www.youtube.com/watch?v=FTr3n7uBIuE
    # 3. Normalization normalizes outputs from convolution.
    # 4. Relu activation:  f(x) = max(0,x). If x < 0, then f(x) = 0.


    model.add(UpSampling2D())
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(256,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(128,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())
    model.add(Conv2D(128,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))


    # Last convolutional layer outputs as many featuremaps as channels 
    # in the final image.
    model.add(Conv2D(channels,kernel_size=3,padding="same"))
    # model.add(Conv2D(channels, kernel_size=1, padding="same"))
    # tanh maps everything to a range between -1 and 1.
    model.add(Activation("tanh"))

    # show the summary of the model architecture
    model.summary()

    # Placeholder for the random noise input
    input = Input(shape=(random_noise_dimension,))
    #Model output
    generated_image = model(input)

    # Change the model type from Sequential to Model (functional API) 
    # More at: https://keras.io/models/model/.
    return Model(input,generated_image)


def build_discriminator(image_shape):
    #Discriminator attempts to classify real and generated images
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, \
                     input_shape=image_shape, padding="same"))
    #Leaky relu is similar to usual relu. If x < 0 then f(x) = x * alpha, 
    # otherwise f(x) = x.
    model.add(LeakyReLU(alpha=0.2))

    # Dropout blocks some connections randomly. This help the model 
    # to generalize better. 0.25 means that every connection has a 25% 
    # chance of being blocked.
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    # Zero padding adds additional rows and columns to the image. 
    # Those rows and columns are made of zeros.
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.25))
    # Flatten layer flattens the output of the previous layer to a single 
    # dimension.
    model.add(Flatten())
    # Outputs a value between 0 and 1 that predicts whether image is 
    # real or generated. 0 = generated, 1 = real.
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    input_image = Input(image_shape)

    #Model output given an image.
    validity = model(input_image)

    return Model(input_image, validity)

def save_images(epoch, random_noise_dimension, generator, target_dir, 
                target_fn, start_epoch = 0):
    #Save generated images for demonstration purposes using matplotlib.pyplot.
    rows, columns = 5, 5
    noise = np.random.normal(0, 1, (rows * columns, random_noise_dimension))
    generated_images = generator.predict(noise)

    generated_images = 0.5 * generated_images + 0.5

    figure, axis = plt.subplots(rows, columns)
    image_count = 0
    for row in range(rows):
        for column in range(columns):
            axis[row,column].imshow(generated_images[image_count, :], \
                                    cmap='spring')
            axis[row,column].axis('off')
            image_count += 1
    figure.savefig(target_dir + target_fn + "_%d.png" % (start_epoch + epoch))
    plt.close()