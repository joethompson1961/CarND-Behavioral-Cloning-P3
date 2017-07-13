import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from keras.models import save_model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import __version__ as keras_version

parser = argparse.ArgumentParser(description='Pretrained model input')
parser.add_argument(
    'model',
    type=str,
    help='Path to model h5 file. Model should be on the same path.'
)
args = parser.parse_args()

src_dir = '../tunedata/'

lines = []
with open(src_dir + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
for line in lines:
    measurement = float(line[3])
    # center image
#    filename = line[0].split('\\')[-1]
    filename = line[0].split('/')[-1]
    path = src_dir + 'IMG/' + filename
    image = cv2.imread(path)
    images.append(image)
    measurements.append(measurement)

    #left image
#    filename = line[1].split('\\')[-1]
    filename = line[1].split('/')[-1]
    path = src_dir + 'IMG/' + filename
    image = cv2.imread(path)
    images.append(image)
    measurements.append(measurement+0.20)  # slight bias to turn right from left view
    
    #right image
#    filename = line[2].split('\\')[-1]
    filename = line[2].split('/')[-1]
    path = src_dir + 'IMG/' + filename
    image = cv2.imread(path)
    images.append(image)
    measurements.append(measurement-0.20)  # slight bias to turn left from right view

X_train = np.array(images)
y_train = np.array(measurements)

# augmented_images = []
# augmented_measurements = []
# for image,measurement in zip(images, measurements):
#     augmented_images.append(image)
#     augmented_measurements.append(measurement)
#     augmented_images.append(cv2.flip(image,1))
#     augmented_measurements.append(measurement * -1.0)
# 
# X_train = np.array(augmented_images)
# y_train = np.array(augmented_measurements)

# model = Sequential()
# model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
# model.add(Cropping2D(cropping=((65,20),(0,0))))
# model = nvidiacar(model)

fn_h5 = args.model + '.h5'
fn_tuned = args.model + '_tuned.h5'

# check that model Keras version is same as local Keras version
f = h5py.File(fn_h5, mode='r')
model_version = f.attrs.get('keras_version')
keras_version = str(keras_version).encode('utf8')
if model_version != keras_version:
    print('You are using Keras version ', keras_version,
          ', but the model was built using ', model_version)

# load model+weights from h5
model = load_model(fn_h5)

# # load json model and model weights from h5
# fn_json = args.model + '.json'
# fn_h5 = args.model + '.h5'
# json_file = open(fn_json, 'r')
# model_json = json_file.read()
# json_file.close()
# model = model_from_json(model_json)
# model.load_weights(fn_h5)

keras.optimizers.RMSprop(lr=0.0001)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2, verbose=1)

# save tuned model - the other way
#model.save_weights(fn_tuned)

# save tuned model
save_model(model, fn_tuned)

# history_object = model.fit_generator(train_generator, samples_per_epoch =
#     len(train_samples), validation_data = 
#     validation_generator,
#     nb_val_samples = len(validation_samples), 
#     nb_epoch=5, verbose=1)

### print the keys contained in the history object
#print(history_object.history.keys())

### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()

exit()
