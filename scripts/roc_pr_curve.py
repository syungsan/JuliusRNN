#!/usr/bin/env python
# coding: utf-8

import os
import codecs
import csv
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


WINDOW_TITLE = "Graph View of JuliusRNN"

# Path
BASE_ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
DATA_DIR_PATH = BASE_ABSOLUTE_PATH + "data"
SETTING_FILE_PATH = DATA_DIR_PATH + "/setting.conf"
GRAPH_DIR_PATH = DATA_DIR_PATH + "/graphs"


def read_csv(file_path, delimiter):

    lists = []
    file = codecs.open(file_path, "r", "utf-8")

    reader = csv.reader(file, delimiter=delimiter)

    for line in reader:
        lists.append(line)

    file.close
    return lists


def roc_curve(y_test, y_preds, model_name):

    if not os.path.isdir(GRAPH_DIR_PATH):
        os.makedirs(GRAPH_DIR_PATH)

    word = read_csv(file_path=SETTING_FILE_PATH, delimiter=",")[0][0]

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_preds)
    auc = metrics.auc(fpr, tpr)
    print("AUC {}".format(auc))

    plt.plot(fpr, tpr, label="ROC curve (area = %.2f)" % auc)
    plt.plot(np.linspace(1, 0, len(fpr)), np.linspace(1, 0, len(fpr)), label="Random ROC curve (area = %.2f)" % 0.5, linestyle="--", color="gray")

    plt.legend()
    plt.title("ROC curve of %s by %s" % (word, model_name))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)

    plt.gcf().canvas.set_window_title(WINDOW_TITLE)

    plt.savefig(GRAPH_DIR_PATH + "/roc_curve_of_%s_by_%s.svg" % (word, model_name), format="svg")
    plt.savefig(GRAPH_DIR_PATH + "/roc_curve_of_%s_by_%s.png" % (word, model_name), format="png")
    plt.savefig(GRAPH_DIR_PATH + "/roc_curve_of_%s_by_%s.pdf" % (word, model_name), format="pdf")

    plt.show()
    plt.close()

    return auc


def pr_curve(y_test, y_preds, model_name):

    if not os.path.isdir(GRAPH_DIR_PATH):
        os.makedirs(GRAPH_DIR_PATH)

    word = read_csv(file_path=SETTING_FILE_PATH, delimiter=",")[0][0]

    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_preds)

    auc = metrics.auc(recall, precision)
    print("AUC {}".format(auc))

    plt.plot(recall, precision, label="PR curve (area = %.2f)" % auc)
    plt.legend()
    plt.title("PR curve of %s by %s" % (word, model_name))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)

    plt.gcf().canvas.set_window_title(WINDOW_TITLE)

    plt.savefig(GRAPH_DIR_PATH + "/pr_curve_of_%s_by_%s.svg" % (word, model_name), format="svg")
    plt.savefig(GRAPH_DIR_PATH + "/pr_curve_of_%s_by_%s.png" % (word, model_name), format="png")
    plt.savefig(GRAPH_DIR_PATH + "/pr_curve_of_%s_by_%s.pdf" % (word, model_name), format="pdf")

    plt.show()
    plt.close()

    return auc
