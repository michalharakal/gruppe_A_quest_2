# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 07:56:39 2020

@author: Wolfgang Reuter

Inspired and altered from:
    https://github.com/platonovsimeon/dcgan-facegenerator
    
"""

# =============================================================================
# Imports
# =============================================================================

# Model functions from keras
from keras.layers import Input, Reshape, Dropout, Dense, Flatten, \
                         BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam

# matplotlib will help with displaying the results
import matplotlib.pyplot as plt
# numpy for some mathematical operations
import numpy as np
# PIL for opening,resizing and saving images
from PIL import Image
# tqdm for a progress bar when loading the dataset
from tqdm import tqdm

#os library is needed for extracting filenames from the dataset folder.
import os

# Custom functions
import gan_functions

# =============================================================================
# Paths and Variables
# =============================================================================

# Data Parameters
imgs_dir = 'data'

image_height, image_width, channels = 64, 64, 3
image_shape = (image_width, image_height, channels)

random_noise_dimension = 100
rows, columns = 4, 8

save_images_interval = 100

target_dir = 'generated_faces'

target_fn = "/generated"

# Model parameters

use_pretrained_model = False

pretrained_model_path_generator = "saved_models/face_generator.h5"
pretrained_model_path_discriminator = "saved_models/face_discriminator.h5"

epochs = 5000
batch_size = 32

start_epoch = 0
if use_pretrained_model:
    assert(start_epoch == 0)

optimizer = Adam(0.0002, 0.5)
optimizer = Adam(0.0002, 0.5)

# TODO: Rewrite get_random_noise() to use batch_size
assert(rows * columns == batch_size)

# =============================================================================
# Functions
# =============================================================================

# A compendium of available functions in gan_functions 

# get_random_noise(rows, columns, random_noise_dimensions)

# build_generator(random_noise_dimension, channels)

# build_discriminator()

# =============================================================================
# Get the training data and normalize it
# =============================================================================

#Get the real images
training_data = gan_functions.get_training_data(imgs_dir, 
                                                image_width,
                                                image_height, 
                                                channels)

#Map all values to a range between -1 and 1.
training_data = training_data / 127.5 - 1.

# =============================================================================
# Set up the labels for generated and real images
# =============================================================================

# Two arrays of labels. Labels for real images: [1,1,1 ... 1,1,1], 
# labels for generated images: [0,0,0 ... 0,0,0]
labels_for_real_images = np.ones((batch_size,1)) - 0.15
labels_for_generated_images = np.zeros((batch_size,1))

# =============================================================================
# Set up generator and discriminator
# =============================================================================

if use_pretrained_model:
    generator = load_model(pretrained_model_path_generator)
    discriminator = load_model(pretrained_model_path_discriminator)
else:
    generator = gan_functions.build_generator(random_noise_dimension, channels)
    discriminator = gan_functions.build_discriminator(image_shape)
    
discriminator.compile(loss="binary_crossentropy", 
                      optimizer=optimizer,
                      metrics=["accuracy"])

# Set up the actual GAN (= combined_model)
random_input = Input(shape=(random_noise_dimension,))
generated_image = generator(random_input)
discriminator.trainable = False
validity = discriminator(generated_image)#
combined_model = Model(random_input, validity) # This is the actual GAN

combined_model.compile(loss="binary_crossentropy",
                       optimizer=optimizer)


# =============================================================================
# Train GAN
# =============================================================================

for epoch in range(epochs):
    # Select a random batch of real images
    indices = np.random.randint(0,training_data.shape[0],batch_size)
    real_images = training_data[indices]
    
    # Generate random noise for a whole batch.
    random_noise = gan_functions.get_random_noise(rows, 
                                                  columns, 
                                                  random_noise_dimension)
    
    discriminator.trainable = True
    
    #Generate a batch of new images.
    generated_images = generator.predict(random_noise)
    
    #Train the discriminator on real images.
    discriminator_loss_real = \
        discriminator.train_on_batch(real_images,
                                     labels_for_real_images)
    #Train the discriminator on generated images.
    discriminator_loss_generated = \
        discriminator.train_on_batch(generated_images,
                                     labels_for_generated_images)
    #Calculate the average discriminator loss.
    discriminator_loss = 0.5 * np.add(discriminator_loss_real,
                                      discriminator_loss_generated)
    
    # Train the generator using the combined model. Generator tries to trick 
    # discriminator into mistaking generated images as real.
    discriminator.trainable = False
    
    labels_for_tricking_discriminator = np.ones((batch_size,1))
    generator_loss = \
        combined_model.train_on_batch(random_noise,labels_for_tricking_discriminator)
    
    # Training ends above (one iteration) 
    # This is only for display and saving models
    print ("%d [Discriminator loss: %f, acc.: %.2f%%] [Generator loss: %f]" % \
           (epoch, discriminator_loss[0], 100*discriminator_loss[1], 
            generator_loss))

    if epoch % save_images_interval == 0:
        gan_functions.save_images(epoch, random_noise_dimension, generator, 
                                  target_dir, target_fn, start_epoch)
        
    #Save the model for a later use
    generator.save(pretrained_model_path_generator)
    discriminator.save(pretrained_model_path_discriminator)

