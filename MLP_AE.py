#! /usr/bin/env python

from keras.layers import Dense, Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Sequential
from keras.models import mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import glob
from skimage.data import imread
import pdb


X_train = []
y_train = []
X_test = []
y_test = []
datagen = []

def prepare_data(input_path, target_path, test_path):
    # Read input, target and test files
    for input_file in glob.glob(input_path + '/*.png'):
        img = imread(input_file)
        img = img.reshape(1, img.shape[0], img.shape[1])
        X_train.append(img)

    for target_file in glob.glob(target_path + '/*.png'):
        img = imread(input_file)
        img = img.reshape(1, img.shape[0], img.shape[1])
        y_train.append(img)

    for test_file in glob.glob(test_path + '/*.png'):
        img = imread(input_file)
        img = img.reshape(1, img.shape[0], img.shape[1])
        X_test.append(img)

    pdb.set_trace()

    # Pre-process data (std normalization - Z-score)
    #datagen = ImageDataGenerator(featurewise_center=True,
    #                             featurewise_std_normalization=True)

    #datagen.fit(X_train)
    #datagen.fit(y_train)
    #datagen.fit(X_test)


def model():
    model = Sequential()
    # border_mode = 'same' does padding
    model.add(Convolution2D(32, 5, 5, activation='relu', border_mode='same', input_shape=(1, 420, 540)))
    # Input at this stage = (32, 420, 540)
    model.add(MaxPooling2D((2, 2), border_mode='same'))
    # inp dim = (32, 210, 270)
    model.add(Convolution2D(16, 5, 5, activation='relu', border_mode='same'))
    # inp dim = (16, 210, 270)
    model.add(MaxPooling2D((2, 2), border_mode='same'))
    # NOTE : max pooling at this layer gives odd dimensions, can create
    # problems)
    # inp dim = (16, 105, 135)
    model.add(Convolution2D(16, 5, 5, activation='relu', border_mode='same'))
    # inp dim = (16, 105, 135)
    model.add(MaxPooling2D((2, 2), border_mode='same'))
    # inp dim = (16, 52, 67)
    model.add(Convolution2D(8, 5, 5, activation='relu', border_mode='same'))
    # inp dim = (8, 52, 67)
    #model.add(MaxPooling2D(2, 2))
    model.add(Convolution2D(8, 5, 5, activation='relu', border_mode='same'))
    # dim = (8, 52, 67)
    model.add(UpSampling2D((2, 2)))
    # dim = (8, 104, 134)
    model.add(ZeroPadding2D(padding=(1,1)))
    # dim = (8, 105, 135)
    model.add(Convolution2D(16, 5, 5, activation='relu', border_mode='same'))
    # dim = (16, 105, 135)
    model.add(UpSampling2D((2, 2)))
    # dim = (16, 210, 270)
    model.add(Convolution2D(32, 5, 5, activation='relu', border_mode='same'))
    # dim = (32, 210, 270)
    model.add(UpSampling2D((2, 2)))
    # dim = (32, 420, 540)
    model.add(Convolution2D(1, 5, 5, activation='relu', border_mode='same'))
    # dim = (1, 420, 540)

    model.compile(loss='mse', optimizer='adam')

    model.fit(X_train, y_train, nb_epoch=25, batch_size=128, shuffle=True)


def main(input_path, target_path, test_path):
    prepare_data(input_path, target_path, test_path)
    model()


if __name__ == "__main__":
    main('/home/shashank/data/kaggle_doc_denoising/train', '/home/shashank/data/kaggle_doc_denoising/train_cleaned', '/home/shashank/data/kaggle_doc_denoising/test')
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    pdb.set_trace()
