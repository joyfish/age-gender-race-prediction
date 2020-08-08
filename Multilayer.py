# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:33:39 2019
@author: magnus

mibh/Multilayer.py at master · Magnusibh/mibh
https://github.com/Magnusibh/mibh/blob/master/Multilayer.py

"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras

DATA_DIR = "../UTKFace/UTKFace"
TRAIN_TEST_SPLIT = 1.0
IM_WIDTH = IM_HEIGHT = 198
ID_GENDER_MAP = {0: 'male', 1: 'female'}
GENDER_ID_MAP = dict((g, i) for i, g in ID_GENDER_MAP.items())
ID_RACE_MAP = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}
RACE_ID_MAP = dict((r, i) for i, r in ID_RACE_MAP.items())

#ID_GENDER_MAP, GENDER_ID_MAP, ID_RACE_MAP, RACE_ID_MAP

def parse_filepath(filepath):
    try:
        path, filename = os.path.split(filepath)
        filename, ext = os.path.splitext(filename)
        age, gender, race, _ = filename.split("_")
        return int(age), ID_GENDER_MAP[int(gender)], ID_RACE_MAP[int(race)]
    except Exception as e:
        print(filepath)
        return None, None, None
    
age = np.array([30.0,35.0,35.0,53.0,45.0,23.0,23.0,24.0,24.0,42.0])
gender = np.array(['male','male','male','male','male','male','male','male','male','male'])
ethni = np.array(['white','others','others','white','white','white','white','indian','indian','white'])    
files_test = glob.glob(os.path.join("C:\\Users\\magnus\\test_pictures", "*.jpg"))    

dictur = {'age':age,'gender':gender,'race':ethni,'file':files_test}
pd_test = pd.DataFrame(dictur)

files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
attributes = list(map(parse_filepath, files))
df = pd.DataFrame(attributes)
df['file'] = files
df.columns = ['age', 'gender', 'race', 'file']
df = df.dropna()

#df = df[(df['age'] > 10) & (df['age'] < 65)]

p = np.random.permutation(len(df))
train_up_to = int(len(df) * TRAIN_TEST_SPLIT)
#train_idx = p[:train_up_to]
train_idx = p
#test_idx = p[train_up_to:]

# split train_idx further into training and validation set
train_up_to = int(train_up_to * 0.7)
train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]
test_idx= valid_idx
df['gender_id'] = df['gender'].map(lambda gender: GENDER_ID_MAP[gender])
df['race_id'] = df['race'].map(lambda race: RACE_ID_MAP[race])

max_age = df['age'].max()

from keras.utils import to_categorical
from PIL import Image

def get_data_generator(df, indices, for_training, batch_size=16):
    images, ages, races, genders = [], [], [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, age, race, gender = r['file'], r['age'], r['race_id'], r['gender_id']
            im = Image.open(file)
            im = im.resize((200, 200))
            im = np.array(im) / 255.0
            images.append(im)
            ages.append(age / max_age)
            races.append(to_categorical(race, len(RACE_ID_MAP)))
            genders.append(to_categorical(gender, 2))
            if len(images) >= batch_size:
                yield np.array(images), [np.array(ages), np.array(races), np.array(genders)]
                images, ages, races, genders = [], [], [], []
        if not for_training:
            break
        
# =============================================================================
#         
#         
# =============================================================================
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool2D
from keras.layers import Conv2D, MaxPooling2D, Input, BatchNormalization, GlobalMaxPool2D
def conv_block(inp, filters=32, bn=True, pool=True):
    _ = Conv2D(filters=filters, kernel_size=(3,3), activation='relu')(inp)
    if bn:
        _ = BatchNormalization()(_)
    if pool:
        _ = MaxPool2D()(_)
    return _
# =============================================================================
# 
# input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
# _ = conv_block(input_layer, filters=32, bn=False, pool=False)
# _ = MaxPool2D(pool_size=(2, 2))(_)
# _ = Conv2D(filters=32, kernel_size=(3,3), activation='relu')(_)
# _ = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(_)
# _ = Dropout(rate=0.2)(_)
# _ = MaxPool2D(pool_size=(2, 2))(_)
# _ = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(_)
# 
# #_ = conv_block(_, filters=32*2)
# #_ = conv_block(_, filters=32*3)
# #_ = conv_block(_, filters=32*4)
# #_ = conv_block(_, filters=32*5)
# #_ = conv_block(_, filters=32*6)
# bottleneck = GlobalMaxPool2D()(_)
# 
# =============================================================================
input_layer = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
_ = conv_block(input_layer, filters=32, bn=False, pool=False)
_ = MaxPool2D(pool_size=(2, 2))(_)
_ = Conv2D(filters=32, kernel_size=(3,3), activation='relu')(_)
_ = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(_)
_ = MaxPool2D(pool_size=(2, 2))(_)
_ = Dropout(rate=0.25)(_)
_ = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(_)
_ = Conv2D(filters=64, kernel_size=(3,3), activation='relu')(_)
_ = MaxPool2D(pool_size=(2, 2))(_)
_ = Dropout(rate=0.25)(_)

bottleneck = Flatten()(_)
# for age calculation
_ = Dense(units=128, activation='relu')(bottleneck)
_ = Dropout(rate=0.2)(_)
age_output = Dense(units=1, activation='sigmoid', name='age_output')(_)

# for race prediction
_ = Dense(units=128, activation='relu')(bottleneck)
_ = Dropout(rate=0.2)(_)
race_output = Dense(units=len(RACE_ID_MAP), activation='softmax', name='race_output')(_)

# for gender prediction
_ = Dense(units=128, activation='relu')(bottleneck)
_ = Dropout(rate=0.2)(_)
gender_output = Dense(units=len(GENDER_ID_MAP), activation='softmax', name='gender_output')(_)

model = Model(inputs=input_layer, outputs=[age_output, race_output, gender_output])
model.compile(optimizer='adadelta', 
              loss={'age_output': 'logcosh', 'race_output': 'categorical_crossentropy', 'gender_output': 'categorical_crossentropy'},
              loss_weights={'age_output': 2., 'race_output': 1.5, 'gender_output': 1.},
              metrics={'age_output': 'mae', 'race_output': 'accuracy', 'gender_output': 'accuracy'})
# model.summary()
# =============================================================================
# 
# 
# =============================================================================
from keras.callbacks import ModelCheckpoint

batch_size = 32
valid_batch_size = 32
train_gen = get_data_generator(df, train_idx, for_training=True, batch_size=batch_size)
valid_gen = get_data_generator(df, valid_idx, for_training=True, batch_size=valid_batch_size)

callbacks = [
    ModelCheckpoint("./model_checkpoint", monitor='val_loss')
]

history = model.fit_generator(train_gen,
                    steps_per_epoch=len(train_idx)//batch_size,
                    epochs=30,
                    callbacks=callbacks,
                    validation_data=valid_gen,
                    validation_steps=len(valid_idx)//valid_batch_size)
# =============================================================================
# =============================================================================
# # 
# =============================================================================
# =============================================================================


def  plot_train_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].plot(history.history['race_output_acc'], label='Ethnicity Train accuracy')
    axes[0].plot(history.history['val_race_output_acc'], label='Ethnicity Val accuracy')
    axes[0].set_xlabel('Epochs')

    
    axes[0].plot(history.history['gender_output_acc'], label='Gender Train accuracy')
    axes[0].plot(history.history['val_gender_output_acc'], label='Gener Val accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].legend()

    axes[1].plot(history.history['age_output_mean_absolute_error'], label='Age Train MAE')
    axes[1].plot(history.history['val_age_output_mean_absolute_error'], label='Age Val MAE')
    axes[1].set_xlabel('Epochs')


    axes[1].plot(history.history['loss'], label='Training loss')
    axes[1].plot(history.history['val_loss'], label='Validation loss')
    axes[1].set_xlabel('Epochs')
    axes[1].legend()
 #   fig.save('epoch30_3layer.jpg')

plot_train_history(history)

# =============================================================================
# 
# =============================================================================

test_gen = get_data_generator(df, test_idx, for_training=False, batch_size=3400)
#dict(zip(model.metrics_names, model.evaluate_generator(test_gen, steps=len(test_idx)//10)))

#test_gen = get_data_generator(df, test_idx, for_training=False, batch_size=10)
x_test, (age_true2, race_true, gender_true)= next(test_gen)
gender_pred = model.predict_on_batch(x_test)


# =============================================================================
# 
# =============================================================================
# =============================================================================
# from sklearn.metrics import classification_report
# print("Classification report for race")
# print(classification_report(race_true, race_pred))
# 
# print("\nClassification report for gender")
# print(classification_report(gender_true, gender_pred))
# =============================================================================

import math
n = 10
random_indices = np.random.permutation(n)
n_cols = 5
n_rows = math.ceil(n / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 20))
for i, img_idx in enumerate(random_indices):
    ax = axes.flat[i]
    ax.imshow(x_test[img_idx])
    ax.set_title('a:{}, g:{}, r:{}'.format(int(age_pred[img_idx]*max_age), ID_GENDER_MAP[np.argmax(gender_pred[img_idx])], ID_RACE_MAP[np.argmax(race_pred[img_idx])]))
    ax.set_xlabel('a:{}, g:{}, r:{}'.format(int(age_true[img_idx]*max_age), ID_GENDER_MAP[np.argmax(gender_true[img_idx])], ID_RACE_MAP[np.argmax(race_true[img_idx])]))
    ax.set_xticks([])
    ax.set_yticks([])
fig.savefig('test_sample30E4L.jpg')
