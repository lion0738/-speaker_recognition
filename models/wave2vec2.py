import glob
import json
import librosa
import numpy as np
import os
import time

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers, models

from dataset import wav_sampling_rate, wav_max_length, gender_list, age_list, dialect_list

model_name = "wav2vec2"
batch_size = 16

def preprocess_dataset(dataset):
    return dataset

# 모델
def get_model():
    input_shape = (240000, )
    title_input = tf.keras.Input(
        shape=input_shape, name="input"
    )
       
    nonsemantic_module = hub.load('https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/3')
    pretrained_layer = nonsemantic_module(samples=wav_as_float_or_int16, sample_rate=16000)['embedding']
    conv = layers.Conv1D(256, (3), activation='relu')(pretrained_layer)
    pool = layers.MaxPooling1D((2))(conv)
    drop = layers.Dropout(0.5)(pool)
    conv = layers.Conv1D(256, (3), activation='relu')(drop)
    pool = layers.MaxPooling1D((2))(conv)
    drop = layers.Dropout(0.5)(pool)
    conv = layers.Conv1D(256, (3), activation='relu')(drop)
    pool = layers.MaxPooling1D((2))(conv)
    drop = layers.Dropout(0.5)(pool)
    flat = layers.Flatten()(pretrained_layer)

    age_dense = layers.Dense(512, activation='relu')(flat)
    age_batch = layers.BatchNormalization()(age_dense)
    age_pred = layers.Dense(len(age_list), activation="softmax", name="age")(age_batch)

    dialect_dense = layers.Dense(512, activation='relu')(flat)
    dialect_batch = layers.BatchNormalization()(dialect_dense)
    dialect_pred = layers.Dense(len(dialect_list), activation="softmax", name="dialect")(dialect_batch)
    
    gender_dense = layers.Dense(512, activation='relu')(flat)
    gender_batch = layers.BatchNormalization()(gender_dense)
    gender_pred = layers.Dense(1, activation="sigmoid", name="gender")(gender_batch)

    model = tf.keras.Model(
        inputs=title_input,
        outputs=[age_pred, dialect_pred, gender_pred],
    )
    
    return model
