from flask import Flask, render_template, request

import numpy as np
import io
import librosa
import scipy.io.wavfile

import ssl
import tensorflow as tf

import models.resnet
import models.efficientnet
import models.efficientnetb5

from pydub import AudioSegment
from models.efficientnet import model_name, batch_size, get_mfccs, calculate_features
from dataset import wav_sampling_rate, wav_max_length, age_list, dialect_list, gender_list

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

@app.route("/")
def show_recorder():
    return render_template("main.html")

@app.route("/upload", methods=['POST'])
def handle_audio():
    features = preprocess_audio(request.files['audio_file'])
    predict = model.predict(features)

    return predict_to_str(predict)

def predict_to_str(predict):
    age = predict[0][0]
    dialect = predict[1][0]
    gender = predict[2][0][0]

    return_str = ""
    return_str += f"\n나이 - {age_list[np.argmax(age)]}\n"
    for age_str, val in zip(age_list, age):
        return_str += f"{age_str} - {round(val, 2)}\n"

    return_str += f"\n지역 - {dialect_list[np.argmax(dialect)]}\n"
    for dialect_str, val in zip(dialect_list, dialect):
        return_str += f"{dialect_str} - {round(val, 2)}\n"

    return_str += f"\n성별 - {'남성' if gender < 0.5 else '여성' }\n"
    return_str += f"gender_predict - {round(gender, 2)}"

    return return_str

def add_zero_padding(audio):
    zero_padding = tf.zeros([wav_sampling_rate * wav_max_length] - tf.shape(audio), dtype=tf.float32)
    audio = tf.cast(audio, tf.float32)
    audio = tf.concat([audio, zero_padding], 0)

    return audio

def parse_audio(audio_file):
    audio = AudioSegment.from_wav(audio_file)
    audio = audio.set_frame_rate(48000)
    audio = audio.set_channels(1)

    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)

    decoded_wav, _ = tf.audio.decode_wav(wav_io.read(), desired_channels=1)

    return decoded_wav

def preprocess_audio(audio_file):
    audio = parse_audio(audio_file)
    #audio, _ = librosa.effects.trim(audio)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cond(tf.shape(audio) < wav_sampling_rate * wav_max_length, lambda: add_zero_padding(audio), lambda: audio)
    audio = tf.image.random_crop(value=audio, size=(wav_sampling_rate * wav_max_length,))

    mfccs, _ = get_mfccs(audio, None)
    features = calculate_features(mfccs)
    features = tf.expand_dims(features, axis=0)

    return features

def get_layer(model, input_layer):
    model_layer = model.get_model()
    model_layer.compile()
    latest_checkpoint_path = os.path.join("checkpoint", model.model_name, "latest.ckpt")
    model_layer.load_weights(latest_checkpoint_path)

    return model_layer(input_layer)

def get_model():
    model_arr = [models.resnet, models.efficientnet, models.efficientnetb5]
    output_names = ["age", "dialect", "gender"]
    input_layer = tf.keras.Input(shape=(498, 32, 3))
    layer_arr = [get_layer(model, input_layer) for model in model_arr]

    #input_reduce_layer = tf.keras.layers.Lambda(lambda inp: inp[:, :, :, 0], output_shape=(498, 32))(input_layer)
    #lstm_layer_arr = [get_model(models.dvector_single, input_reduce_layer)]
    #layer_arr.extend(lstm_layer_arr)

    output_layer = [tf.keras.layers.Average(name=output_names[i])([output[i] for output in layer_arr]) for i in range(3)]

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    #model.compile()

    model.compile()
    
    return model

if __name__ == "__main__":
    model = get_model()
    #model.compile(metrics="accuracy")
    #model.load_weights("latest.ckpt")

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)
    ssl_context.load_cert_chain(certfile="ssl/certificate.crt", keyfile="ssl/private.key")
    app.run(host="0.0.0.0", port=443, ssl_context=ssl_context, debug=False)
