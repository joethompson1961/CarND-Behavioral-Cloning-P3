import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import argparse
import json
import sklearn
import re
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.models import save_model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import __version__ as keras_version

def lenet(model):
    model.add(Convolution2D(6, 5, 5, activation="relu", padding='same'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Convolution2D(6, 5, 5, activation="relu", padding='same'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def nvidiacar(model):    
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu", border_mode='valid'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu", border_mode='valid'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu", border_mode='valid'))
    model.add(Convolution2D(64, 3, 3, activation="relu", border_mode='valid'))
    model.add(Convolution2D(64, 3, 3, activation="relu", border_mode='valid'))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

# returns batches of images for training
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for sample in batch_samples:
                name = sample[0]
                image = cv2.imread(name)
                images.append(image)
                angle = float(sample[1])
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

parser = argparse.ArgumentParser(description='Model trainer')
parser.add_argument(
    'model',
    type=str,
    help='name of h5 model file (without .h5 extension).'
)
args = parser.parse_args()

# Reading in training data log file
#src_dir = 'C:/Udacity/SDCND/term1/resources/training-data/behavior-cloning/simdata'
src_dir = 'C:/Udacity/SDCND/term1/resources/training-data/behavior-cloning/lessdata'
#src_dir = 'C:/Udacity/SDCND/term1/resources/training-data/behavior-cloning/data'
samples = []
with open(src_dir + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Create flattened/augmented training data log with compensated steering angles for
# left, center and right cameras.
flattened_samples = []
for sample in samples:
    angle = float(sample[3])
    
    # support windows and linux file path formats
    sample[0] = sample[0].replace('\\','/')
    sample[1] = sample[1].replace('\\','/')
    sample[2] = sample[2].replace('\\','/')
    
    # center image
    name = src_dir + '/IMG/' + sample[0].split('/')[-1]
    flattened_samples.append([name, angle])

    # left image
    name = src_dir + '/IMG/' + sample[1].split('/')[-1]
    flattened_samples.append([name, (angle + 0.20)])

    # right image
    name = src_dir + '/IMG/' + sample[2].split('/')[-1]
    flattened_samples.append([name, (angle - 0.20)]) # compensate angle to steer slightly left for right camera.

# Split the data log into training and validation datasets
train_samples, validation_samples = train_test_split(flattened_samples, test_size=0.2)
n_train = len(train_samples)
print("n_train:", n_train)
n_valid = len(validation_samples)
print("n_valid:", n_valid)

# Use generators for training the model - generators use memory efficiently to support larger datasets.
train_generator = generator(train_samples, batch_size=256)
validation_generator = generator(validation_samples, batch_size=256)

# Define the model architecture:
# - normilize the input data
# - crop input data images to focus on meaningful portion
# - use nvidia car architecture, it has worked well for this project
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3))) # normalize input data to range [-0.5,+0.5]
model.add(Cropping2D(cropping=((65,20), (0,0)))) # crop top 40% (sky) and bottom 12% (hood) from input image data
model = nvidiacar(model)  # use the nvidia car architecture

# Compile and train  the model.
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
history_object = model.fit_generator(train_generator, samples_per_epoch = n_train,
                                     validation_data=validation_generator, nb_val_samples = n_valid,
                                     nb_epoch=1, verbose = 1)

# save model + weights
fn_h5 = args.model + '.h5'
save_model(model, fn_h5)
 
# # alternative: save json model and model weights
# fn_json = args.model + '.json'
# fn_h5 = args.model + '.h5'
# model_json = model.to_json()
# with open(fn_json, "w") as json_file:
#     json_file.write(model_json)
# model.save_weights(fn_h5)
 
# set keras version attribute of the weights file
f = h5py.File(fn_h5, mode='r+')
f.attrs.modify('keras_version', str(keras_version).encode('utf8'))
f.close()

# print the keys contained in the history object
print(history_object.history.keys())

# plot training metrics
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(history_object.history['loss'])
ax1.plot(history_object.history['val_loss'])
ax1.set_title('model mean squared error loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('mean squared error loss')
ax1.legend(['training set', 'validation set'], loc='upper right')
ax2.plot(history_object.history['acc'])
ax2.plot(history_object.history['val_acc'])
ax2.set_title('model accuracy')
ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')
ax2.legend(['training set', 'validation set'], loc='upper right')
plt.subplots_adjust(left=0.12, right=0.88, top=0.9, bottom=0.15)
plt.show()

exit()