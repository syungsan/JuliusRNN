#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras.models import load_model
import os
import csv
import codecs
import scipy.stats as sp
import glob
import shutil

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import roc_pr_curve as roc_pr

import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


FEATURE_MAX_LENGTH = 40

ENSEMBLE_NUMBER = 5

# MODEL_NAMES = ["SimpleRNN", "SimpleRNNStack", "LSTM", "GRU", "Bidirectional_LSTM", "Bidirectional_GRU", "BiLSTMStack", "BiGRUStack", "CNN_RNN_BiLSTM", "CNN_RNN_BiGRU"]
MODEL_NAMES = ["SimpleRNNStack"]

# Path
BASE_ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
DATA_DIR_PATH = BASE_ABSOLUTE_PATH + "data"
TEST_FILE_PATH = DATA_DIR_PATH + "/test.csv"
LOG_DIR_PATH = DATA_DIR_PATH + "/logs"
SETTING_FILE_PATH = DATA_DIR_PATH + "/setting.conf"
BEST_MODELS_FILE_PATH = LOG_DIR_PATH + "/ensemble_model.csv"


def read_csv(file_path, delimiter):

    lists = []
    file = codecs.open(file_path, "r", 'utf-8')

    reader = csv.reader(file, delimiter=delimiter)

    for line in reader:
        lists.append(line)

    file.close
    return lists


def write_csv(file_path, list):

    try:
        # 書き込み UTF-8
        with open(file_path, "a", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
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


if __name__ == "__main__":

    setting = read_csv(file_path=SETTING_FILE_PATH, delimiter=",")
    time_series_max_length = len(setting[0][2])

    if not os.path.exists(BEST_MODELS_FILE_PATH):
        best_model_labels = [["model_name", "ensemble_number", "max_accuracy", "min_loss", "precision", "recall", "f1", "roc_auc", "pr_auc", "best_model_name"]]
        write_csv(BEST_MODELS_FILE_PATH, best_model_labels)

    X, y = load_data(file_name=TEST_FILE_PATH)

    # 時系列に変換
    X = np.reshape(X, (X.shape[0], time_series_max_length, FEATURE_MAX_LENGTH), order="F")

    # 3次元配列の2次元要素を1次元に（iOSのための取り計らい）
    X = np.reshape(X, (X.shape[0], (time_series_max_length * FEATURE_MAX_LENGTH)))

    # Z-Score関数による正規化
    X = sp.stats.zscore(np.array(X), axis=1)

    y = y.tolist()

    for model_name in MODEL_NAMES:
        for ensemble_number in range(ENSEMBLE_NUMBER):

            model_dir_path = LOG_DIR_PATH + "/" + model_name + "/ensemble_{}".format(ensemble_number + 1)

            # データが多すぎるので10飛ばしに絞る
            final_model_paths = glob.glob("/{}/*_final_*.h5".format(model_dir_path))
            pre_process_model_paths = glob.glob("/{}/*_pre_process_model_*.h5".format(model_dir_path))

            model_paths = []
            model_paths.extend(final_model_paths)

            for index in range(10, len(pre_process_model_paths), 10):
                model_paths.append([s for s in pre_process_model_paths if "pre_process_model_*-{}".format("{0:02d}".format(index + 1)) in s])

            print(model_paths)

            print("\n")
            print("Model type name : %s and Ensemble number : %s" % (model_name, ensemble_number + 1))
            print("\n")

            losss = []
            corrects = []

            for model_path in model_paths:

                # 学習済みのモデルを読み込む
                model = load_model(model_path)

                # model.summary()

                model.compile(loss="binary_crossentropy", optimizer="adadelta", metrics=["accuracy"])

                score = model.evaluate(X, y, verbose=0)

                print("\nModel Detail Name : {}\n".format(os.path.basename(model_path)))
                print("Test loss :", score[0])
                print("Test accuracy :", score[1])

                losss.append(score[0])
                corrects.append(score[1])

            max_accuracy = max(corrects)
            max_acc_indexs = [i for i, v in enumerate(corrects) if v == max_accuracy]

            min_losss = []
            for max_acc_index in max_acc_indexs:
                min_losss.append(losss[max_acc_index])

            min_loss = min(min_losss)
            best_model_indexs = []

            for max_acc_index in max_acc_indexs:

                if losss[max_acc_index] == min_loss:
                    best_model_indexs.append(max_acc_index)

            for best_model_index in best_model_indexs:

                best_model_path = model_paths[best_model_index]
                best_model_name = os.path.basename(best_model_path)
                print("\n最大Accuracy = %f ; at loss = %f => %s" % (max_accuracy * 100, losss[best_model_index], best_model_name))

                # best_ensemble_models_dir_path = LOG_DIR_PATH + "/" + model_name

                # if not os.path.isdir(best_ensemble_models_dir_path):
                #     os.makedirs(best_ensemble_models_dir_path)

                best_ensemble_model_path = LOG_DIR_PATH + "/" + model_name + "/ensemble/" + best_model_name
                shutil.copy(best_model_path, best_ensemble_model_path)

                model = load_model(best_model_path)

                # we'll use binary xent for the loss, and AdaDelta as the optimizer
                model.compile(loss="binary_crossentropy", optimizer="adadelta", metrics=["accuracy"])
                print("\n")

                predictions = model.predict(X)

                y_preds = []
                y_probs = []
                index = 0

                for target in y:

                    y_prob = predictions[index][0]
                    y_probs.append(y_prob)

                    if round(y_prob) == target:
                        y_preds.append(1)
                        print(str(y_prob) + " : " + str(target) + " => correct")
                    else:
                        y_preds.append(0)
                        print(str(y_prob) + " : " + str(target) + " => incorrect")

                    index += 1

                print("\n")
                cm = confusion_matrix(y, y_preds)
                print(cm)

                precision = precision_score(y, y_preds)
                recall = recall_score(y, y_preds)
                f1 = f1_score(y, y_preds)

                print("\n")
                print("Precision", precision)
                print("Recall", recall)
                print("F1Score", f1)

                print("\n")
                print("Plot the ROC Curve")
                roc_auc = roc_pr.roc_curve(y, y_probs, "{}_ensemble_{}".format(model_name, ensemble_number + 1))

                print("\n")
                print("Plot the PR Curve")
                pr_auc = roc_pr.pr_curve(y, y_probs, "{}_ensemble_{}".format(model_name, ensemble_number + 1))
                print("\n")

                best_model_results = [[model_name, ensemble_number, max_accuracy * 100, losss[best_model_index], precision, recall, f1, roc_auc, pr_auc, best_model_name]]
                write_csv(BEST_MODELS_FILE_PATH, best_model_results)

    print("")
    print("\nAll process completed...")
