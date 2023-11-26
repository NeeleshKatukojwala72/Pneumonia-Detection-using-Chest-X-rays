# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 22:50:43 2019

@author: DSP
"""
from PIL import Image
import PIL
PIL.__version__

import tensorflow 
import keras
print(tensorflow.__version__)
#from tensorflow.python.framework import ops
#ops.reset_default_graph()

# Building the CNN
# Importing the Keras libraries and packages'''
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

import os
os.getcwd()
#os.chdir()
os.chdir('D:\\major project\\IMAGES\\')

training_set = train_datagen.flow_from_directory('Train_data',target_size = (64, 64),
                                                 batch_size = 32,class_mode = 'binary')#class_mode=categorical to work on multiple

test_set = test_datagen.flow_from_directory('Test_data',target_size = (64, 64),
                                            batch_size = 32,class_mode = 'binary')
# this will take long time

classifier.fit_generator(training_set,
                         steps_per_epoch = 200,
                         nb_epoch = 1, # nb_epoch = 25,
                         validation_data = test_set,
                         validation_steps = 60)#verbose=0

"""
Epoch 1/1
250/250 [==============================] - 2519s 10s/step - loss: 0.1952 - acc: 0.9109 
- val_loss: 1.4666 - val_acc: 0.5417
"""

# In[]
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('validation_data/Malignant/ISIC_0014688.jpg', target_size = (64, 64))
# This is test img
# first arg is the path
# img is 64x64 dims this is what v hv used in training so wee need to use exactly the same dims
# here also
test_image
# In[ ]:
test_image = image.img_to_array(test_image)
# Also in our first layer below it is a 3D array
# Step 1 - Convolution
# classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
# this will convert from a 3D img to 3D array
test_image # shld gv us (64,64,3)
test_image = np.expand_dims(test_image, axis = 0)
# axis specifies the position of indx of the dimnsn v r addng
# v need to add the dim in the first position
test_image # now it shld show (1,64,64,3)
result = classifier.predict(test_image)
# v r trying to predict
result # gv us 1
# In[ ]:
print(training_set.class_indices)

if result[0][0] == 0:
    prediction = 'Benign'
else:
    prediction = 'Malignant'

print(prediction)




# In[ ]:
#import numpy as np
#from keras.preprocessing import image
test_image = image.load_img('validation_data/Benign/ISIC_0000686.jpg', target_size = (64, 64))

print(prediction)

