import glob
import json
import librosa
import numpy as np
import os
import time

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers, models

from dataset import wav_sampling_rate, wav_max_length, gender_list, age_list, dialect_list

model_name = "dvector-bidirection"
batch_size = 8

lower_edge_hertz, upper_edge_hertz, num_mel_bins, num_mfccs = 40.0, 8000.0, 80, 13
frame_length_ms, frame_step_overlap = 0.025, 0.4

frame_length = int(wav_sampling_rate * frame_length_ms)
frame_step = int(frame_length * frame_step_overlap)
fft_length = frame_length

def get_mfccs(waveform, labels):
    
    stfts = tf.signal.stft(waveform, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)

    spectrograms = tf.abs(stfts)

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, wav_sampling_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)[..., :num_mfccs]

    return mfccs, labels

def resize_mfccs(mfccs, labels):
    mfccs = tf.expand_dims(mfccs, axis=-1)
    mfccs = tf.image.resize(mfccs, (224, 224), method="nearest")
    mfccs = tf.squeeze(mfccs, axis=-1)

    return mfccs, labels

def calculate_features(waveform):
    #audio = librosa.feature.melspectrogram(y=audio.numpy(), sr=wav_sampling_rate, n_fft=fft_length, hop_length=frame_step, win_length=frame_length)
    #mfccs = librosa.feature.mfcc(y=audio, sr=wav_sampling_rate, _mfcc=num_mfccs)
    mfccs = librosa.feature.mfcc(y=waveform.numpy(), sr=wav_sampling_rate, n_mfcc=num_mfccs, win_length=frame_length, hop_length=frame_step, htk=True)
    delta = librosa.feature.delta(mfccs)
    delta_delta = librosa.feature.delta(mfccs, order=2)

    return tf.concat([mfccs, delta, delta_delta], axis=-2)

def get_features(mfccs, labels):
    features = tf.py_function(calculate_features, inp=[mfccs], Tout=tf.float32)
    return features, labels

def preprocess_dataset(dataset):
    dataset = dataset.map(get_mfccs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #dataset = dataset.map(resize_mfccs, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #dataset = dataset.map(get_features, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset

# 모델
def get_model():
    input_shape = (int((wav_sampling_rate * wav_max_length - frame_length) / frame_step) + 1, num_mfccs)
    title_input = tf.keras.Input(
        shape=input_shape, name="input"
    )

    lstm = layers.Bidirectional(layers.LSTM(1024, return_sequences=True))(title_input)
    lstm = layers.Bidirectional(layers.LSTM(1024, return_sequences=True))(lstm)
    lstm = layers.Bidirectional(layers.LSTM(1024))(lstm)
    dense = layers.Dense(1024, activation='relu')(lstm)

    age_dense = layers.Dense(2048, activation='relu')(dense)
    age_batch = layers.BatchNormalization()(age_dense)
    age_pred = layers.Dense(len(age_list), activation="softmax", name="age")(age_batch)

    dialect_dense = layers.Dense(2048, activation='relu')(dense)
    dialect_batch = layers.BatchNormalization()(dialect_dense)
    dialect_pred = layers.Dense(len(dialect_list), activation="softmax", name="dialect")(dialect_batch)
    
    gender_dense = layers.Dense(2048, activation='relu')(dense)
    gender_batch = layers.BatchNormalization()(gender_dense)
    gender_pred = layers.Dense(1, activation="sigmoid", name="gender")(gender_batch)

    model = tf.keras.Model(
        inputs=title_input,
        outputs=[age_pred, dialect_pred, gender_pred],
    )
    
    return model
