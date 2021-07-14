#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import sqlite3
import os
import codecs
import csv
import shutil


WINDOW_TITLE = "Graph View of JuliusRNN"

# Path
BASE_ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
DATA_DIR_PATH = BASE_ABSOLUTE_PATH + "data"
DATABASE_FILE_PATH = DATA_DIR_PATH + "/evaluation_single.sqlite3"
SETTING_FILE_PATH = DATA_DIR_PATH + "/setting.conf"
GRAPH_DIR_PATH = DATA_DIR_PATH + "/graphs"


def learning_curve(datas, line_names, title, xlabel, ylabel, text, text_y_pos):

    plt.style.use("ggplot")

    for i in range(0, len(datas)):
        x = range(len(datas[i]))
        plt.plot(x, datas[i], label=line_names[i], linewidth=3, linestyle="solid") # dashdot

    # pylab.ylim(0.0, 1.2)

    # ラベルの追加
    plt.title(title, fontsize=20) # タイトル
    plt.ylabel(ylabel, fontsize=20) # Y 軸
    plt.xlabel(xlabel, fontsize=20) # X 軸

    plt.text(0.1, text_y_pos, text)

    # 凡例
    # pylab.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend()

    # グリッド有効
    # pylab.grid(True)

    # ウィンドウタイトル
    plt.gcf().canvas.set_window_title(WINDOW_TITLE)

    # svgに保存
    plt.savefig(GRAPH_DIR_PATH + "/" + title.replace(" ", "_") + ".svg", format="svg")
    # pngに保存
    plt.savefig(GRAPH_DIR_PATH + "/" + title.replace(" ", "_") + ".png", format="png")
    # pdfに保存
    plt.savefig(GRAPH_DIR_PATH + "/" + title.replace(" ", "_") + ".pdf", format="pdf")

    # 描画
    plt.show()

    # プロットを表示するためにポップアップしたウィンドウをクローズ
    # このプロジェクトのみの対処方法
    plt.close()


def get_from_database(database_file_path, sql):

    db = sqlite3.connect(database_file_path)
    cur = db.cursor()
    cur.execute(sql)

    output = []
    for row in cur:
        output.append(row[0])

    cur.close()
    db.close()

    return output


def read_csv(file_path, delimiter):

    lists = []
    file = codecs.open(file_path, "r", "utf-8")

    reader = csv.reader(file, delimiter=delimiter)

    for line in reader:
        lists.append(line)

    file.close
    return lists


if __name__ == "__main__":

    # if os.path.isdir(GRAPH_DIR_PATH):
    #     shutil.rmtree(GRAPH_DIR_PATH)

    if not os.path.isdir(GRAPH_DIR_PATH):
        os.makedirs(GRAPH_DIR_PATH)

    word = read_csv(file_path=SETTING_FILE_PATH, delimiter=",")[0][0]

    # プロットするレコードを設定
    # models = ["SimpleRNN", "SimpleRNNStack", "LSTM", "GRU", "Bidirectional_LSTM", "Bidirectional_GRU", "BiLSTMStack", "BiGRUStack", "CNN_RNN_BiLSTM","CNN_RNN_BiGRU", "QRNN"]
    models = ["SimpleRNN", "SimpleRNNStack", "LSTM", "Bidirectional_LSTM", "BiLSTMStack", "CNN_RNN_BiLSTM"]

    # プロットするカラムを設定
    results = ["training_accuracy", "validation_accuracy", "training_loss", "validation_loss"]

    # グラフのタイトル
    titles = ["binary train accuracy of ", "binary valid accuracy of ", "binary train loss of ", "binary valid loss of "]

    # yラベル
    ylabels = ["Accuracy", "Accuracy", "Loss", "Loss"]

    datas = []
    for i in range(len(results)):

        line = []
        for j in range(len(models)):

            sql = "SELECT %s FROM learning WHERE model = '%s';" % (results[i], models[j])
            data = get_from_database(DATABASE_FILE_PATH, sql)

            # NoneをListから削除
            cut_data = [item for item in data if item is not None]

            line.append(cut_data)

        datas.append(line)

    for k in range(len(results)):

        title = titles[k] + word + " Single"
        rate = 0.0
        index = 0
        model = ""
        text = ""

        if k == 0 or k == 1:

            l = 0
            max_datas = []
            max_factors = []

            for data in datas[k]:

                # maxのindexを全返し
                # [m for m, x in enumerate(data) if x == max(data)]

                max_datas.append(max(data))
                max_factors.append([max(data), [i for i, x in enumerate(data) if x == max(data)][0], models[l]])

                if k == 0:
                    print("Max Train Accuracy => model : %s ; value : %f ; index : %d" % (models[l], max(data), data.index(max(data))))
                if k == 1:
                    print("Max Valid Accuracy => model : %s ; value : %f ; index : %d" % (models[l], max(data), data.index(max(data))))

                l += 1

            rate = max_factors[[i for i, x in enumerate(max_datas) if x == max(max_datas)][0]][0]
            index = max_factors[[i for i, x in enumerate(max_datas) if x == max(max_datas)][0]][1]
            model = max_factors[[i for i, x in enumerate(max_datas) if x == max(max_datas)][0]][2]

            text = "max " + " = " + str(rate * 100.0) + "%, " + model + ", " + str(index) + "epoch"

        if k == 2 or k == 3:

            l = 0
            min_datas = []
            min_factors = []

            for data in datas[k]:

                min_datas.append(min(data))
                min_factors.append([min(data), [i for i, x in enumerate(data) if x == min(data)][0], models[l]])

                if k == 2:
                    print("Min train loss => model : %s ; value : %f ; index : %d" % (models[l], min(data), data.index(min(data))))
                if k == 3:
                    print("Min train loss => model : %s ; value : %f ; index : %d" % (models[l], min(data), data.index(min(data))))

                l += 1

            rate = min_factors[[i for i, x in enumerate(min_datas) if x == min(min_datas)][0]][0]
            index = min_factors[[i for i, x in enumerate(min_datas) if x == min(min_datas)][0]][1]
            model = min_factors[[i for i, x in enumerate(min_datas) if x == min(min_datas)][0]][2]

            text = "min " + " = " + str(rate) + "%, " + model + ", " + str(index) + "epoch"

        learning_curve(datas[k], models, title, "Epoch", ylabels[k], text, rate)

    print("")
    print("\nAll process completed...")
