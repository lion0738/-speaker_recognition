import os

import tensorflow as tf
import tensorflow_datasets as tfds

SEED = 42

wav_sampling_rate = 48000
wav_max_length = 5

gender_list = ["Male", "Female"]
age_list = ["11~19", "20~29", "30~39", "40~49", "50~59", "over60"]
dialect_list = ["경기/서울", "경상", "전라", "충청", "강원", "제주"]

def load_database(train_sql, test_sql):
    sql_output = (tf.string, tf.string, tf.string, tf.string)

    train_ds = tf.data.experimental.SqlDataset("sqlite", "database/training.db", train_sql, sql_output)
    val_ds = tf.data.experimental.SqlDataset("sqlite", "database/validation.db", train_sql, sql_output)
    test_ds = tf.data.experimental.SqlDataset("sqlite", "database/test.db", test_sql, sql_output)

    train_ds = train_ds.map(preprocess_database, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(preprocess_database, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(preprocess_database, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
    val_ds = val_ds.apply(tf.data.experimental.ignore_errors())
    test_ds = test_ds.apply(tf.data.experimental.ignore_errors())

    return train_ds, val_ds, test_ds

def filter_dataset(data_dict):
    is_over_60 = tf.math.logical_or(data_dict["age"] == "60~69", data_dict["age"] == "over70")
    data_dict["age"] = tf.cond(is_over_60, lambda: "over60", lambda: data_dict["age"])

    age_in_list = tf.math.reduce_any(data_dict["age"] == age_list)
    dialect_in_list = tf.math.reduce_any(data_dict["dialect"] == dialect_list)
    return tf.math.logical_and(age_in_list, dialect_in_list)

def load_dataset(name="voice_dialect_ds"):
    train_ds, val_ds, test_ds = tfds.load(name, \
            split=["train[:80%]", "train[80%:]", "test"], \
            shuffle_files=True)

    train_ds = train_ds.filter(filter_dataset)
    val_ds = val_ds.filter(filter_dataset)
    test_ds = test_ds.filter(filter_dataset)

    train_ds = train_ds.map(preprocess_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(preprocess_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(preprocess_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
    val_ds = val_ds.apply(tf.data.experimental.ignore_errors())
    test_ds = test_ds.apply(tf.data.experimental.ignore_errors())

    return train_ds, val_ds, test_ds

@tf.function
def add_zero_padding(audio):
    zero_padding = tf.zeros([wav_sampling_rate * wav_max_length] - tf.shape(audio), dtype=tf.float32)
    audio = tf.cast(audio, tf.float32)
    audio = tf.concat([audio, zero_padding], 0)

    return audio

@tf.function
def get_audio_and_labels(audio, gender, age, dialect):
    audio, _ = tf.audio.decode_wav(audio, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cond(tf.shape(audio) < wav_sampling_rate * wav_max_length, lambda: add_zero_padding(audio), lambda: audio)
    audio = tf.image.random_crop(value=audio, size=(wav_sampling_rate * wav_max_length,), seed=SEED)

    is_over_60 = tf.math.logical_or(age == "60~69", age == "over70")
    age = tf.cond(is_over_60, lambda: "over60", lambda: age)

    gender_id = tf.argmax(gender == gender_list)
    age_id = tf.one_hot(tf.argmax(age == age_list), len(age_list))
    dialect_id = tf.one_hot(tf.argmax(dialect == dialect_list), len(dialect_list))

    return audio, {"gender": gender_id, "age": age_id, "dialect": dialect_id}

@tf.function
def preprocess_database(file_name, gender, age, dialect):
    audio_file_name = tf.strings.regex_replace(file_name, ".json", ".wav")
    audio = tf.io.read_file(audio_file_name)

    return get_audio_and_labels(audio, gender, age, dialect)

@tf.function
def preprocess_dataset(data_dict):
    return get_audio_and_labels(data_dict["audio"], data_dict["gender"], data_dict["age"], data_dict["dialect"])
