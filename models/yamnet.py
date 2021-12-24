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

model_name = "yamnet"
batch_size = 16

def calculate_features(audio):
    audio = librosa.resample(audio.numpy(), wav_sampling_rate, 16000)

    return audio

def get_features(mfccs, labels):
    features = tf.py_function(calculate_features, inp=[mfccs], Tout=tf.float32)
    
    return features, labels

def preprocess_dataset(dataset):
    dataset = dataset.map(get_features, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset

# 모델
def get_model():
    input_shape = (16000 * wav_max_length)
    title_input = tf.keras.Input(
        shape=input_shape, name="input"
    )

    model = tf.saved_model.load("yamnet_1")(title_input)
    flat = layers.Flatten()(model)

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
