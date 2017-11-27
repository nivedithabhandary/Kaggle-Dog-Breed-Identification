#!/usr/bin/env python
from __future__ import print_function
import fnmatch
import glob
import numpy as np
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import time

import keras
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.applications.vgg19 import VGG19, preprocess_input

img_train_dir = "/train"
label_file = '/labels.csv'
img_val_dir = "/val"
img_test_dir = "/test"
# lets set average width and height here
# https://www.kaggle.com/jeru666/dog-eat-dog-world-eda-useful-scripts
img_width, img_height = 443//5, 386//5
train_model=True
use_image_data_generator = True
seed_model=None # set this to actual model file

def gen_data_file(img_files, id_label_dict=None, num_class=None):
    """ Gerantes data using img and label id dictionary
    """
    X = []
    y = []

    for idx, img_file in enumerate(img_files):
        img = load_img(img_file, target_size = (img_width, img_height))
        x = img_to_array(img)

        # randomly flip images per epoch
        if random.choice([True, False]):
            x = np.fliplr(x)

        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        X.append(x)
        if id_label_dict is not None:
            y.append(one_hot_encode(num_class, id_label_dict[os.path.basename(img_file).split(".jpg")[0]]))
        else:
            y.append(0)

    X = np.vstack(X)
    y = np.vstack(y)
    return X, y

def gen_label_dict(label_file, label_dict=None):
    """Generate label dictionary"""
    if label_dict is None:
        label_dict = {}
    id_label_dict = {}
    i = len(label_dict.keys())
    df =  pd.read_csv(label_file)
    df_breed = df['breed']
    df_id = df['id']

    for idx, breed in enumerate(df_breed):
        if breed not in label_dict.keys():
            label_dict[breed] = i
            i += 1
        id_label_dict[df_id[idx]] = label_dict[breed]

    return label_dict, id_label_dict

def one_hot_encode(size, pos):
    one_hot = np.zeros(size)
    np.put(one_hot, pos, 1)
    return one_hot

def gen_img_files(img_dir, shuffle=True):
    """ Generates list of image files
    """
    img_file_path = []
    img_files = fnmatch.filter(os.listdir(img_dir),  '*.jpg')
    if shuffle is True:
        random.shuffle(img_files)
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        img_file_path.append(img_path)

    return img_file_path

def gen_data_dir(img_dir, id_label_dict, num_class, shuffle=True):
    """ Takes img directory and id_label dict
    """
    img_file_path = gen_img_files(img_dir, shuffle)
    return gen_data_file(img_file_path, id_label_dict, num_class)

def gen_batch(img_dir, id_label_dict, batch_size, num_class, shuffle=True):
    """ Generates batches of data
    """
    img_file_path = gen_img_files(img_dir, shuffle)
    num_images = len(img_file_path)
    while True:
        for i in range(0, num_images-batch_size, batch_size):
            X, y = gen_data_file(img_file_path[i:i+batch_size], id_label_dict, num_class)
            yield X, y

def get_model(num_class):
    """ Get vgg19 model for regression experiment
    """
    # import VGG model and use it
    model = VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3), pooling='max')

    # Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
    for layer in model.layers:
        layer.trainable = False

    #Adding custom Layers
    x = model.output
    #x = Flatten()(x)
    #x = Dropout(0.5)(x)
    x = Dense(120, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(120, activation="relu")(x)
    predictions = Dense(num_class, activation='softmax')(x)

    # This is the model we will train
    model = Model(inputs=model.input, outputs=predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, verbose=1)]
    print(model.summary())

    return model

def get_label_idx_name(label_dict):
    label_idx = []
    label_name = []
    for breed in label_dict.keys():
        label_name.append(breed)
        label_idx.append(label_dict[breed])

    return np.array(label_idx), label_name

if __name__ == '__main__':
    #main()
    batch_size = 32
    epochs = 500
    label_dict, id_label_dict = gen_label_dict(label_file)
    num_class = len(label_dict.keys())
    print('num_class is ', num_class)

    if train_model is True:
        if seed_model is None:
            model = get_model(num_class)
        else:
            model = load_model(seed_model)
            print(model.summary())

        checkpoint = ModelCheckpoint('dog_breed.h5', monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

        # get validation data
        print('generating validation data')
        x_val, y_val = gen_data_dir(img_val_dir, id_label_dict, num_class)
        xy_val = (x_val, y_val)

        img_files = gen_img_files(img_train_dir)
        num_train_samples = len(img_files)
        print('num train images', num_train_samples)

        if use_image_data_generator is True:
            print ('generating training data for image generator')
            x_train, y_train = gen_data_dir(img_train_dir, id_label_dict, num_class)
            generator = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.2, height_shift_range=0.2, rotation_range=45, shear_range=0.2, zoom_range=0.2, fill_mode='nearest').flow(x_train, y_train)
        else:
            generator = gen_batch(img_train_dir, id_label_dict, batch_size, num_class)

        # Train the model
        print ('fitting model')
        model.fit_generator(
        generator = generator,
        epochs = epochs,
        steps_per_epoch = int(num_train_samples / batch_size) + 100,
        validation_data = xy_val,
        callbacks = [checkpoint, early])

        score = model.evaluate(x_val, y_val)

    model = load_model('dog_breed.h5')
    label_idx, label_name = get_label_idx_name(label_dict)
    test_img_file_paths = gen_img_files(img_test_dir, False)
    num_test_img_files = len(test_img_file_paths)
    out_file = 'submissions/submission_' + str(int(time.time())) + '.csv'
    out_file = open(out_file, 'w')
    out_file.write("id,")
    out_file.write(','.join([l for l in label_name]) + '\n')

    for idx in range(0, num_test_img_files, batch_size):
        end_idx = idx + batch_size
        if end_idx > num_test_img_files:
            end_idx = num_test_img_files

        X, _ = gen_data_file(test_img_file_paths[idx:end_idx])
        prediction = model.predict(X)

        for idx, test_img_file in enumerate(test_img_file_paths[idx:end_idx]):
            image_id = os.path.basename(test_img_file).split(".jpg")[0]
            pred_per_id = prediction[idx][label_idx]
            print (idx, image_id, max(pred_per_id), np.argmax(pred_per_id))
            out_file.write(image_id+',')
            out_file.write(','.join([str(i) for i in pred_per_id]) + '\n')

    out_file.close()
