import pandas as pd
import numpy as np
import csv
from scipy import ndimage

# Setup Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D


meta_data = []
column_data = []
with open('/opt/data/driving_log.csv') as file:
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
data = []
for row in meta_data:
    filename = row[0].split('/')[-1]
    #print(filename)
    file_path = '/opt/data/IMG/' + filename
    #print(file_path)
    #images.append(cv2.imread(file_path))
    images.append(ndimage.imread(file_path))
    data.append(row[3])

#print(images)
#print(data)
print("File Read!")

X_train = np.array(images)
y_train = np.array(data)
#print(y_train)

# TODO: Build the Fully Connected Neural Network in Keras Here
model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))
#model.add(Activation('relu'))
#model.add(Dense(5))
#model.add(Activation('softmax'))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=100)
model.save('model.h5')


