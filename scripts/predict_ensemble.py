#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import math
import scipy.stats as sp
from keras.models import load_model
import shutil
import glob

import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

import pyaudio  # 録音機能を使うためのライブラリ
import wave     # wavファイルを扱うためのライブラリ


# 特徴量の次元数
FEATURE_MAX_LENGTH = 40

# 1モーラの時系列分割数
MORA_TIME_DIVISION_NUMBER = 1

GET_MFCC_METHOD = "aubio"

# models = ["SimpleRNN", "SimpleRNNStack", "LSTM", "GRU", "Bidirectional_LSTM", "Bidirectional_GRU", "BiLSTMStack", "BiGRUStack", "CNN_RNN_BiLSTM", "CNN_RNN_BiGRU", "QRNN"]
MODEL_NAME = "SimpleRNNStack"

# correctness = ["不正解": 0, "正解": 1]
CORRECTNESS = 0

# Path
BASE_ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
TEMP_DIR_PATH = BASE_ABSOLUTE_PATH + "temp"
DATA_DIR_PATH = BASE_ABSOLUTE_PATH + "data"
SETTING_FILE_PATH = DATA_DIR_PATH + "/setting.conf"
LAB_FILE_PATH = TEMP_DIR_PATH + "/predict.lab"
WORD_FILE_PATH = TEMP_DIR_PATH + "/predict.txt"
LOG_DIR_PATH = DATA_DIR_PATH + "/logs"

# Command
SEGMENTATION_COMMAND = BASE_ABSOLUTE_PATH + "perl/bin/perl.exe " + BASE_ABSOLUTE_PATH + "scripts/segment_julius.pl"

RECORD_SECONDS = 3.0 # 録音する時間の長さ（秒）
WAVE_OUTPUT_FILE_PATH = TEMP_DIR_PATH + "/predict.wav" # 音声を保存するファイル名

iDeviceIndex = 0 # 録音デバイスのインデックス番号

# 基本情報の設定
FORMAT = pyaudio.paInt16 # 音声のフォーマット
CHANNELS = 1             # モノラル
RATE = 16000            # サンプルレート
CHUNK = 2**11            # データ点数


def recording():

    audio = pyaudio.PyAudio() # pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        input_device_index = iDeviceIndex, # 録音デバイスのインデックス番号
                        frames_per_buffer=CHUNK)

    #--------------録音開始---------------

    print("recording...")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)


    print("finished recording")

    #--------------録音終了---------------

    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILE_PATH, "wb")
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


if __name__ == "__main__":

    if not os.path.isdir(TEMP_DIR_PATH):
        os.makedirs(TEMP_DIR_PATH)

    import feature as feat
    import output_ensemble as output

    setting = feat.read_csv(file_path=SETTING_FILE_PATH, delimiter=",")
    time_series_max_length = len(setting[0][2])

    model_paths = glob.glob("{}/{}/ensemble/*.h5".format(LOG_DIR_PATH, MODEL_NAME))

    models = []
    for model_path in model_paths:

        model = load_model(model_path)
        model.compile(loss="binary_crossentropy", optimizer="adadelta", metrics=["accuracy"])
        models.append(model)

    recording()

    word = setting[0][1]

    file = open(WORD_FILE_PATH, "w", encoding="utf-8")
    file.write(word + "\n")
    file.close()

    feat.execute_segmentation()

    scores = feat.pick_score(LAB_FILE_PATH, time_series_max_length)
    scores = [float(s) for s in scores]

    frames = feat.pick_frame(LAB_FILE_PATH, time_series_max_length)

    if GET_MFCC_METHOD == "librosa":
        input_mfccs = feat.get_mfcc_librosa(WAVE_OUTPUT_FILE_PATH)

    elif GET_MFCC_METHOD == "aubio":
        input_mfccs = feat.get_mfcc_aubio(WAVE_OUTPUT_FILE_PATH)

    frame2ds = []
    for i in range(int(len(frames) / 2)):
        frame2ds.append([frames[i * 2], frames[i * 2 + 1]])

    features = []
    for input_mfcc in input_mfccs:

        mora_in_section_averages = []
        for frame2d in frame2ds:

            # juliusの解像度は0.01[s]（10[ms]）なのでフレーム数は100をかけて復元
            start_frame = math.floor(float(frame2d[0]) * 100.0)
            end_frame = math.floor(float(frame2d[1]) * 100.0)

            # 1音素を分割しそれぞれの区間平均をとる
            mora_in_section_averages.append(feat.section_average(input_mfcc[start_frame: end_frame], MORA_TIME_DIVISION_NUMBER))

        features.append(mora_in_section_averages)

    # ケツにセグメントのスコア追加
    features.append(scores)

    features = feat.flatten_with_any_depth(features)
    X = np.array([features])

    # 時系列に変換
    X = np.reshape(X, (X.shape[0], time_series_max_length, FEATURE_MAX_LENGTH), order="F")

    # 3次元配列の2次元要素を1次元に（iOSのための取り計らい）
    X = np.reshape(X, (X.shape[0], (time_series_max_length * FEATURE_MAX_LENGTH)))

    # Z-Score関数による正規化
    X = sp.stats.zscore(np.array(X), axis=1)

    correctness, probability = output.output_ensemble(models, X)

    if probability >= 0.5:
        diff_prob = ((probability / 0.5) - 1) * 100.0
        print("Result {}% : => correct".format(str(diff_prob)))
    else:
        diff_prob = (1 - (probability / 0.5)) * 100.0
        print("Result {}% : => incorrect".format(str(diff_prob)))

    if os.path.isdir(TEMP_DIR_PATH):
        shutil.rmtree(TEMP_DIR_PATH)

    print("")
    print("\nAll process completed...")
