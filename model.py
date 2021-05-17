import pandas as pd
import numpy as np
import csv
import cv2
from scipy import ndimage

# Setup Keras
from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D,  Convolution2D
from keras.layers.pooling import MaxPooling2D

import os
path1 = os.path.normpath("data1/")

meta_data = []
column_data = []
with open(path1+'/driving_log.csv') as file:
    #print(file)
    csvreader = csv.reader(file, delimiter=',')
    for i, row in enumerate(csvreader):
        #print(row)
        if i ==0:
            column_data = row
        else:
            meta_data.append(row)
        
          
file.close()
#print(meta_data)
#print(column_data)

images = []
measurements = []
correction = 0.2
for row in meta_data:
    measurements.append(float(row[3]))
    measurements.append((float(row[3])+correction))
    measurements.append((float(row[3])-correction))
    for fname in row[:3]:
        filename = fname.split('/')[-1]
        #print(filename)
        file_path = path1+'/IMG/' + filename
        #print(file_path)
        #images.append(cv2.imread(file_path))
        images.append(ndimage.imread(file_path))

#print(images)
#print(data)
#print("File Read!")

agumented_images=[]
agumented_measurements = []
for image, measurement in zip(images, measurements):
    agumented_images.append(image)
    agumented_measurements.append(measurement)
    agumented_images.append(cv2.flip(image,1))
    agumented_measurements.append(measurement*-1)
    

#X_train = np.array(images)
#y_train = np.array(measurements)
X_train = np.array(agumented_images)
y_train = np.array(agumented_measurements)
#print(y_train)

# TODO: Build the Fully Connected Neural Network in Keras Here
# model = Sequential()
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
#model.add(Flatten()))#(Flatten(input_shape=(160, 320, 3)))
#model.add(Dense(1))
#model.add(Activation('relu'))
#model.add(Activation('softmax'))

# model = Sequential()
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# model.add(Cropping2D(cropping=((70,25),(0,0))))
# model.add(Convolution2D(6,5,5, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6,5,5,activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24,5,5, activation='relu'))
model.add(Convolution2D(36,5,5,activation='relu'))
model.add(Convolution2D(48,5,5,activation='relu'))

model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)#, verbose=2)
model.save('model.h5')


