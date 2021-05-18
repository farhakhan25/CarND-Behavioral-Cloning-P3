import pandas as pd
import numpy as np
import csv
import cv2
from scipy import ndimage
from sklearn.model_selection import train_test_split
import sklearn

# Setup Keras
from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import os


path_strings = ["data1"]  # ["data1", "train_data1"]
paths = [os.path.normpath(string) for string in path_strings]

meta_data = []
column_data = []
data_index = []
ind = 0
for s, path in enumerate(paths):
    with open(path +'/driving_log.csv', 'r') as file:
        csvreader = csv.reader(file)
        for i, row in enumerate(csvreader):
            if i ==0:
                column_data = row
            else:
                meta_data.append(row)
                ind +=1
    data_index.append(ind)


file.close()
print(f"total Data {data_index}")

images = []
measurements = []
correction = 0.25
k= 0
for i, row in enumerate(meta_data):
    measurements.append(float(row[3]))
    measurements.append((float(row[3])+correction))
    measurements.append((float(row[3])-correction))
    if i >= data_index[k]:
        k += 1
    for fname in row[:3]:
        if '/' in fname:
            filename = fname.split('/')[-1]
        elif '\\' in fname:
            filename = fname.split('\\')[-1]
        file_path = paths[k] + '/IMG/' + filename
        images.append(ndimage.imread(file_path))


agumented_images = []
agumented_measurements = []
for image, measurement in zip(images, measurements):
    agumented_images.append(image)
    agumented_measurements.append(measurement)
    agumented_images.append(cv2.flip(image, 1))
    agumented_measurements.append(measurement * -1)

# X_train = np.array(images)
# y_train = np.array(measurements)
X_train = np.array(agumented_images)
y_train = np.array(agumented_measurements)

# #Model 0
# model = Sequential()
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
# model.add(Flatten())  # (Flatten(input_shape=(160, 320, 3)))
# model.add(Dense(1))
# model.add(Activation('relu'))
# model.add(Activation('softmax'))

# #Model 3
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

# #Model 4
# model = Sequential()
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# model.add(Cropping2D(cropping=((70,25),(0,0))))

# model.add(Convolution2D(24,5,5, activation='relu'))
# model.add(Convolution2D(36,5,5,activation='relu'))
# model.add(Convolution2D(48,5,5,activation='relu'))

# model.add(Convolution2D(64,3,3,activation='relu'))
# model.add(Convolution2D(64,3,3,activation='relu'))

# model.add(Flatten())
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(1))

# #Model 5
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(24,5,5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(36,5,5,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(48,5,5,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.20))
model.add(Dense(10))
model.add(Dense(1))

def _generator(X_samples, y_samples, batch=32):
    num_samples = len(X_samples)
    while 1:
        for offset in range(0, num_samples, batch):
            X_data = X_samples[offset:offset+batch]
            y_data = y_samples[offset:offset + batch]
            yield sklearn.utils.shuffle(X_data, y_data)

X_train_samples, X_validation_samples, y_train_samples, y_validation_samples = train_test_split(X_train, y_train, test_size=0.2, random_state=0, shuffle=True)

batch=50
train_generator = _generator(X_train_samples, y_train_samples, batch=batch)
validation_generator = _generator(X_validation_samples, y_validation_samples, batch=batch)

model.compile(loss='mse', optimizer='adam')
print(len(X_train_samples))
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)  # , verbose=2)
model.fit_generator(train_generator, steps_per_epoch=np.ceil(len(X_train_samples)/batch),validation_data=validation_generator, validation_steps=np.ceil(len(X_validation_samples)/batch), epochs=5, verbose=2)
print(model.summary())
model.save('model.h5')

