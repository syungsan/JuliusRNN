#!/usr/bin/env python
# coding: utf-8

# cuda ver10.1
# pip install tensorflow==2.2.*
import numpy as np
# ↓ pip install keras==2.3.*
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, GRU, SimpleRNN
from keras.layers import Bidirectional, MaxPooling1D, Conv1D, Reshape, Dropout, Flatten
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping # , TensorBoard
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE # pip install imbalanced-learn
import scipy.stats as sp
import os
import csv
import sqlite3
import codecs
import datetime

# for QRNN
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.layers import SpatialDropout1D
from scripts.qrnn import QRNN

import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


# 特徴量の次元数（MFCC: 39 + Score: 1）
FEATURE_MAX_LENGTH = 40

IS_SMOTE = True

# Path
BASE_ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
DATA_DIR_PATH = BASE_ABSOLUTE_PATH + "data"
TRAINING_FILE_PATH = DATA_DIR_PATH + "/train.csv"
LOG_DIR_PATH = DATA_DIR_PATH + "/logs"
DATABASE_PATH = DATA_DIR_PATH + "/evaluation_single.sqlite3"
SETTING_FILE_PATH = DATA_DIR_PATH + "/setting.conf"

# for Cross Validation Evaluation Method
# クロスバリデーションの分割数
# {reference equation: k = log(n) / log(2)}
FOLDS_NUMBER = 10

# EARLY_STOPPING_PATIENCE = 0

# Hyper Parameter
RNN_HIDDEN_NEURONS = 64 # 32 # RNN層のユニット数 default 20
BATCH_SIZE = 32 # [1, 8, 16, 32, 64, 128, 256, 512]
EPOCHS = 500 # 100 # 学習回数

# MODEL_NAMES = ["SimpleRNN", "SimpleRNNStack", "LSTM", "GRU", "Bidirectional_LSTM", "Bidirectional_GRU", "BiLSTMStack", "BiGRUStack", "CNN_RNN_BiGRU", "CNN_RNN_BiLSTM", "QRNN"]
MODEL_NAMES = ["SimpleRNN", "SimpleRNNStack", "LSTM", "Bidirectional_LSTM", "BiLSTMStack", "CNN_RNN_BiLSTM"]


def read_csv(file_path, delimiter):

    lists = []
    file = codecs.open(file_path, "r", "utf-8")

    reader = csv.reader(file, delimiter=delimiter)

    for line in reader:
        lists.append(line)

    file.close
    return lists


def load_data(file_path):
    # load your data using this function

    # CSVの制限を外す
    # csv.field_size_limit(sys.maxsize)

    data = []
    target = []

    with open(file_path, "r") as f:
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


def create_model(model_name, time_series_max_length):
    # create your model using this function

    model = None

    if model_name == "SimpleRNN":

        model = Sequential()

        model.add(Reshape((time_series_max_length, FEATURE_MAX_LENGTH), input_shape=(time_series_max_length * FEATURE_MAX_LENGTH,)))
        model.add(SimpleRNN(RNN_HIDDEN_NEURONS, dropout=0.1, recurrent_dropout=0.5))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        model.compile(loss="binary_crossentropy",
                      optimizer="adadelta",
                      metrics=["accuracy"])

    if model_name == "SimpleRNNStack":

        model = Sequential()

        model.add(Reshape((time_series_max_length, FEATURE_MAX_LENGTH), input_shape=(time_series_max_length * FEATURE_MAX_LENGTH,)))
        model.add(SimpleRNN(RNN_HIDDEN_NEURONS, return_sequences=True))
        model.add(SimpleRNN(RNN_HIDDEN_NEURONS, dropout=0.1, recurrent_dropout=0.5))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        model.compile(loss="binary_crossentropy",
                      optimizer="adadelta",
                      metrics=["accuracy"])

    if model_name == "LSTM":

        model = Sequential()

        model.add(Reshape((time_series_max_length, FEATURE_MAX_LENGTH), input_shape=(time_series_max_length * FEATURE_MAX_LENGTH,)))
        model.add(LSTM(RNN_HIDDEN_NEURONS, dropout=0.1, recurrent_dropout=0.5))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        model.compile(loss="binary_crossentropy",
                      optimizer="adadelta",
                      metrics=["accuracy"])

    if model_name == "GRU":

        model = Sequential()

        model.add(Reshape((time_series_max_length, FEATURE_MAX_LENGTH), input_shape=(time_series_max_length * FEATURE_MAX_LENGTH,)))
        model.add(GRU(RNN_HIDDEN_NEURONS, dropout=0.1, recurrent_dropout=0.5))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        model.compile(loss="binary_crossentropy",
                      optimizer="adadelta",
                      metrics=["accuracy"])

    if model_name == "Bidirectional_LSTM":

        model = Sequential()

        model.add(Reshape((time_series_max_length, FEATURE_MAX_LENGTH), input_shape=(time_series_max_length * FEATURE_MAX_LENGTH,)))
        model.add(Bidirectional(LSTM(RNN_HIDDEN_NEURONS, dropout=0.1, recurrent_dropout=0.5)))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        model.compile(loss="binary_crossentropy",
                      optimizer="adadelta",
                      metrics=["accuracy"])

    if model_name == "Bidirectional_GRU":

        model = Sequential()

        model.add(Reshape((time_series_max_length, FEATURE_MAX_LENGTH), input_shape=(time_series_max_length * FEATURE_MAX_LENGTH,)))
        model.add(Bidirectional(GRU(RNN_HIDDEN_NEURONS, dropout=0.1, recurrent_dropout=0.5)))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        model.compile(loss="binary_crossentropy",
                      optimizer="adadelta",
                      metrics=["accuracy"])

    if model_name == "BiLSTMStack":

        # ２層のLSTMベースの双方向RNN（ + dropout + recurrent_dropout）
        model = Sequential()

        model.add(Reshape((time_series_max_length, FEATURE_MAX_LENGTH), input_shape=(time_series_max_length * FEATURE_MAX_LENGTH,)))
        model.add(Bidirectional(LSTM(RNN_HIDDEN_NEURONS, return_sequences=True)))
        model.add(Bidirectional(LSTM(RNN_HIDDEN_NEURONS, dropout=0.1, recurrent_dropout=0.5)))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        model.compile(loss="binary_crossentropy",
                      optimizer="adadelta",
                      metrics=["accuracy"])

    if model_name == "BiGRUStack":

        # ２層のGRUベースの双方向RNN（ + dropout + recurrent_dropout）
        model = Sequential()

        model.add(Reshape((time_series_max_length, FEATURE_MAX_LENGTH), input_shape=(time_series_max_length * FEATURE_MAX_LENGTH,)))
        model.add(Bidirectional(GRU(RNN_HIDDEN_NEURONS, return_sequences=True)))
        model.add(Bidirectional(GRU(RNN_HIDDEN_NEURONS, dropout=0.1, recurrent_dropout=0.5)))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        model.compile(loss="binary_crossentropy",
                      optimizer="adadelta",
                      metrics=["accuracy"])

    if model_name == "CNN_RNN_BiGRU":

        model = Sequential()

        model.add(Reshape((time_series_max_length, FEATURE_MAX_LENGTH), input_shape=(time_series_max_length * FEATURE_MAX_LENGTH,)))
        model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=2, padding="same"))
        model.add(Dropout(0.3))
        model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=2, padding="same"))
        model.add(Dropout(0.35))
        model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=2, padding="same"))
        model.add(Dropout(0.4))

        model.add(Bidirectional(GRU(RNN_HIDDEN_NEURONS, return_sequences=True, dropout=0.1, recurrent_dropout=0.5)))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.45))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        model.compile(loss="binary_crossentropy",
                      optimizer="adadelta",
                      metrics=["accuracy"])

    if model_name == "CNN_RNN_BiLSTM":

        model = Sequential()

        model.add(Reshape((time_series_max_length, FEATURE_MAX_LENGTH), input_shape=(time_series_max_length * FEATURE_MAX_LENGTH,)))
        model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=2, padding="same"))
        model.add(Dropout(0.3))
        model.add(Conv1D(filters=64, kernel_size=3, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=2, padding="same"))
        model.add(Dropout(0.35))
        model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=2, padding="same"))
        model.add(Dropout(0.4))

        model.add(Bidirectional(LSTM(RNN_HIDDEN_NEURONS, return_sequences=True, dropout=0.1, recurrent_dropout=0.5)))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.45))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        model.compile(loss="binary_crossentropy",
                      optimizer="adadelta",
                      metrics=["accuracy"])

    if model_name == "QRNN":

        model = Sequential()

        model.add(Reshape((time_series_max_length, FEATURE_MAX_LENGTH), input_shape=(time_series_max_length * FEATURE_MAX_LENGTH,)))
        model.add(SpatialDropout1D(0.2, input_shape=(time_series_max_length, FEATURE_MAX_LENGTH)))

        model.add(QRNN(RNN_HIDDEN_NEURONS, window_size=3, dropout=0.2,
                       kernel_regularizer=l2(1e-4), bias_regularizer=l2(1e-4),
                       kernel_constraint=maxnorm(10), bias_constraint=maxnorm(10)))

        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        # try using different optimizers and different optimizer configs
        model.compile(loss="binary_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

    return model


if __name__ == "__main__":

    # if os.path.isdir(LOG_DIR_PATH):
    #     shutil.rmtree(LOG_DIR_PATH)

    if not os.path.isdir(LOG_DIR_PATH):
        os.makedirs(LOG_DIR_PATH)

    setting = read_csv(file_path=SETTING_FILE_PATH, delimiter=",")
    time_series_max_length = len(setting[0][2])

    # オリジナル特徴量の流入
    X, y = load_data(file_path=TRAINING_FILE_PATH)

    if IS_SMOTE:

        # num_y = []
        # for i in range(2):
        #     num_y.append(np.sum(y == i))

        # max_num_y = max(num_y)

        # _ratio = {0: max_num_y, 1: max_num_y}

        # SMOTE
        # smote = SMOTE(sampling_strategy=_ratio, random_state=71)

        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)

        for i in range(2):
            print("Data number resampled => " + str(i) + ": " + str(np.sum(y == i)))

    # 時系列に変換
    X = np.reshape(X, (X.shape[0], time_series_max_length, FEATURE_MAX_LENGTH), order="F")

    # 3次元配列の2次元要素を1次元に（iOSのための取り計らい）
    X = np.reshape(X, (X.shape[0], (time_series_max_length * FEATURE_MAX_LENGTH)))

    # Z-Score関数による正規化
    X = sp.stats.zscore(X, axis=1)

    # Cross Validation Evaluation Method

    # define X-fold cross validation
    kf = StratifiedKFold(n_splits=FOLDS_NUMBER, shuffle=True)
    mdl = 0

    train_acc = []
    valid_acc = []
    train_loss = []
    valid_loss = []

    for model_name in MODEL_NAMES:

        cvscores = []

        fld = 0
        train_acc_d = [[0 for i in range(FOLDS_NUMBER)] for j in range(EPOCHS)]
        valid_acc_d = [[0 for i in range(FOLDS_NUMBER)] for j in range(EPOCHS)]
        train_loss_d = [[0 for i in range(FOLDS_NUMBER)] for j in range(EPOCHS)]
        valid_loss_d = [[0 for i in range(FOLDS_NUMBER)] for j in range(EPOCHS)]

        # cross validation
        for train, test in kf.split(X, y):

            model_proccess = mdl / len(MODEL_NAMES) * 100
            fold_proccess = 100 / len(MODEL_NAMES) * fld / FOLDS_NUMBER

            print("\n")
            print("Proccess was " + str(int(model_proccess + fold_proccess)) + "% completed.")
            print("Model Name is " + model_name + ".")
            print("Running Fold", fld + 1, "/", FOLDS_NUMBER)

            model = None # Clearing the NN.
            model = create_model(model_name, time_series_max_length)

            if not os.path.isdir(LOG_DIR_PATH + "/" + model_name):
                os.makedirs(LOG_DIR_PATH + "/" + model_name)

            now = datetime.datetime.now()

            csv_logger = CSVLogger(LOG_DIR_PATH + "/" + model_name + "/log_acc_loss.csv", append=False)
            fpath = LOG_DIR_PATH + "/" + model_name + "/" + model_name + "_{0:%Y-%m-%d}".format(now) + "_single_pre_process_model_" + "{0:02d}".format(fld + 1) + "-{epoch:02d}-{loss:.2f}-{val_loss:.2f}-{accuracy:.2f}-{val_accuracy:.2f}.h5"
            # cp_cb = ModelCheckpoint(filepath=fpath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
            model_checkpoint = ModelCheckpoint(filepath=fpath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='min', period=0)
            # es_cb = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, verbose=1, mode='auto')
            # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=EARLY_STOPPING_PATIENCE, verbose=1, mode='auto')
            # tensorboard = TensorBoard(log_dir=LOG_DIR_PATH + "/" + model_name, histogram_freq=0, write_graph=True)

            hist = model.fit(X[train], y[train], validation_data=(X[test], y[test]), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[model_checkpoint]) # , early_stopping]) # , tensorboard])

            # Evaluate
            scores = model.evaluate(X[test], y[test], verbose=0)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)

            for i in model.layers:
                if type(i) is Dropout:
                    model.layers.remove(i)

            # Save the model
            model.save(LOG_DIR_PATH + "/" + model_name + "/" + model_name + "_{0:%Y-%m-%d}".format(now) + "_single_final_{}_model_".format(EPOCHS) + "{0:02d}".format(fld + 1) + "_" + str(round(scores[1] * 100, 1)) + "%_model.h5")

            for epoch in range(0, EPOCHS):
                train_acc_d[epoch][fld] = hist.history["accuracy"][epoch]
                valid_acc_d[epoch][fld] = hist.history["val_accuracy"][epoch]
                train_loss_d[epoch][fld] = hist.history["loss"][epoch]
                valid_loss_d[epoch][fld] = hist.history["val_loss"][epoch]

            fld += 1

        train_acc_d_means = []
        valid_acc_d_means = []
        train_loss_d_means = []
        valid_loss_d_means = []

        for epoch in range(0, EPOCHS):
            train_acc_d_means.append(np.mean(train_acc_d[epoch]))
            valid_acc_d_means.append(np.mean(valid_acc_d[epoch]))
            train_loss_d_means.append(np.mean(train_loss_d[epoch]))
            valid_loss_d_means.append(np.mean(valid_loss_d[epoch]))

        train_acc.append(train_acc_d_means)
        valid_acc.append(valid_acc_d_means)
        train_loss.append(train_loss_d_means)
        valid_loss.append(valid_loss_d_means)

        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

        mdl += 1

    if os.path.exists(DATABASE_PATH):
        os.remove(DATABASE_PATH)

    db = sqlite3.connect(DATABASE_PATH)
    cur = db.cursor()

    sql = "CREATE TABLE IF NOT EXISTS learning (model TEXT, epoch INTEGER, training_accuracy REAL, validation_accuracy REAL, training_loss REAL, validation_loss REAL);"

    cur.execute(sql)
    db.commit()

    datas = []
    for i in range(0, len(MODEL_NAMES)):
        for j in range(len(hist.epoch)):

            # なぜか3番目のtrain accuracyがバイナリーになるのでfloatにキャストし直し（sqliteのバージョンによるバグ？）
            datas.append((MODEL_NAMES[i], j + 1, float(train_acc[i][j]), valid_acc[i][j], train_loss[i][j], valid_loss[i][j]))

    sql = "INSERT INTO learning (model, epoch, training_accuracy, validation_accuracy, training_loss, validation_loss) VALUES (?, ?, ?, ?, ?, ?);"

    cur.executemany(sql, datas)
    db.commit()

    cur.close()
    db.close()

    print("\n")
    print("All Proccess was completed.")
    print("\n")
