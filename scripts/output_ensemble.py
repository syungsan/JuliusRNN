#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras.models import load_model
import os
import csv
import codecs
import scipy.stats as sp
import glob
from statistics import mean

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import roc_pr_curve as roc_pr

import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


FEATURE_MAX_LENGTH = 40

# Path
BASE_ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
DATA_DIR_PATH = BASE_ABSOLUTE_PATH + "data"
TEST_FILE_PATH = DATA_DIR_PATH + "/test.csv"
LOG_DIR_PATH = DATA_DIR_PATH + "/logs"
SETTING_FILE_PATH = DATA_DIR_PATH + "/setting.conf"
ENSEMBLE_MODEL_DETAIL_FILE_PATH = LOG_DIR_PATH + "/ensemble_model.csv"


def read_csv(file_path, delimiter):

    lists = []
    file = codecs.open(file_path, "r", "utf-8")

    reader = csv.reader(file, delimiter=delimiter)

    for line in reader:
        lists.append(line)

    file.close
    return lists


def write_csv(file_path, list):

    try:
        # 書き込み UTF-8
        with open(file_path, "a", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, lineterminator="\n")
            writer.writerows(list)

    # 起こりそうな例外をキャッチ
    except FileNotFoundError as e:
        print(e)
    except csv.Error as e:
        print(e)


def load_data(file_name):
    # load your data using this function

    # CSVの制限を外す
    # csv.field_size_limit(sys.maxsize)

    data = []
    target = []

    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter=",")

        for columns in reader:
            target.append(columns[0])

            # こいつが決め手か？！
            data.append(columns[1:])

    data = np.array(data, dtype=np.float32)

    # なぜか進数のエラーを返すので処理
    target10b = []
    for tar in target:
        target10b.append(int(float(tar)))

    target = np.array(target10b, dtype=np.int32)

    return data, target


def output_ensemble(models, x):

    correctness = []
    probabilitys = []

    for index, model in enumerate(models):

        predictions = model.predict(x)

        threshold = 0.50
        probability = predictions[0][0]

        if probability >= threshold:
            print("model_{} output correct".format(index + 1))
            correctness.append(1)
        else:
            print("model_{} output incorrect".format(index + 1))
            correctness.append(0)

        probabilitys.append(probability)

    if correctness.count(1) >= 3:
        correct = 1
    else:
        correct = 0

    return correct, mean(probabilitys)


if __name__ == "__main__":

    ENSEMBLE_NUMBER = 5

    # MODEL_NAMES = ["SimpleRNN", "SimpleRNNStack", "LSTM", "GRU", "Bidirectional_LSTM", "Bidirectional_GRU", "BiLSTMStack", "BiGRUStack", "CNN_RNN_BiLSTM", "CNN_RNN_BiGRU"]
    MODEL_NAMES = ["SimpleRNNStack"]

    setting = read_csv(file_path=SETTING_FILE_PATH, delimiter=",")
    time_series_max_length = len(setting[0][2])

    write_csv(ENSEMBLE_MODEL_DETAIL_FILE_PATH, [[]])
    write_csv(ENSEMBLE_MODEL_DETAIL_FILE_PATH, [["ensemble result"]])

    best_ensemble_labels = [["model_name", "max_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
    write_csv(ENSEMBLE_MODEL_DETAIL_FILE_PATH, best_ensemble_labels)

    X, y = load_data(file_name=TEST_FILE_PATH)

    # 時系列に変換
    X = np.reshape(X, (X.shape[0], time_series_max_length, FEATURE_MAX_LENGTH), order="F")

    # 3次元配列の2次元要素を1次元に（iOSのための取り計らい）
    X = np.reshape(X, (X.shape[0], (time_series_max_length * FEATURE_MAX_LENGTH)))

    # Z-Score関数による正規化
    X = sp.stats.zscore(np.array(X), axis=1)

    Y = y.tolist()

    for model_name in MODEL_NAMES:

        model_paths = glob.glob("{}/{}/ensemble/*.h5".format(LOG_DIR_PATH, model_name))

        models = []
        for model_path in model_paths:

            model = load_model(model_path)
            model.compile(loss="binary_crossentropy", optimizer="adadelta", metrics=["accuracy"])

            models.append(model)

        correctnesss = []
        probabilitys = []

        for x, y in zip(X, Y):

            x = np.array([x])

            correctness, probability = output_ensemble(models, x)

            if correctness == y:
                correctnesss.append(1)
                print(str(probability) + " : " + str(y) + " => correct")
            else:
                correctnesss.append(0)
                print(str(probability) + " : " + str(y) + " => incorrect")

            probabilitys.append(probability)

        accuracy = correctnesss.count(1) / len(Y) * 100

        print("\n")
        print("Accuracy {}".format(accuracy))

        print("\n")
        cm = confusion_matrix(Y, correctnesss)
        print(cm)

        precision = precision_score(Y, correctnesss)
        recall = recall_score(Y, correctnesss)
        f1 = f1_score(Y, correctnesss)

        print("\n")
        print("Precision", precision)
        print("Recall", recall)
        print("F1Score", f1)

        print("\n")
        print("Plot the ROC Curve")
        roc_auc = roc_pr.roc_curve(Y, probabilitys, "{}_all_ensemble".format(model_name))

        print("\n")
        print("Plot the PR Curve")
        pr_auc = roc_pr.pr_curve(Y, probabilitys, "{}_all_ensemble".format(model_name))
        print("\n")

        best_ensemble_results = [[model_name, accuracy, precision, recall, f1, roc_auc, pr_auc]]
        write_csv(ENSEMBLE_MODEL_DETAIL_FILE_PATH, best_ensemble_results)

    print("")
    print("\nAll process completed...")
