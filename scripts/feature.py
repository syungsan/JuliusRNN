#!/usr/bin/env python
# coding: utf-8

import os
import sys
import glob
import csv
import codecs
import subprocess
import math
import shutil
import numpy as np
from statistics import mean
import scipy.signal
import librosa
import librosa.feature
from aubio import source, pvoc, mfcc, filterbank


# 1モーラの時系列分割数
MORA_TIME_DIVISION_NUMBER = 1

# Training Only = 1, Training & Test = 2
METHOD_PATTERN = 2

# Select calc MFCC method. ["librosa", "aubio"]
GET_MFCC_METHOD = "aubio"

# Path
BASE_ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
TEMP_DIR_PATH = BASE_ABSOLUTE_PATH + "temp"
DATA_DIR_PATH = BASE_ABSOLUTE_PATH + "data"
TRAINING_WAV_DIR_PATHS = [DATA_DIR_PATH + "/wavs/bad_wav", DATA_DIR_PATH + "/wavs/ok_wav"]
TEST_WAV_DIR_PATHS = [DATA_DIR_PATH + "/wavs/test/bad_wav", DATA_DIR_PATH + "/wavs/test/ok_wav"]
SETTING_FILE_PATH = DATA_DIR_PATH + "/setting.conf"
TRAINING_FILE_PATH = DATA_DIR_PATH + "/train.csv"
TEST_FILE_PATH = DATA_DIR_PATH + "/test.csv"

# Label
CLASS_LABELS = [0, 1]
WAV_LABELS = ["bad", "ok"]

# Command
SEGMENTATION_COMMAND = BASE_ABSOLUTE_PATH + "perl/bin/perl.exe " + BASE_ABSOLUTE_PATH + "scripts/segment_julius.pl"

# 正誤各waveファイルの最大保持設定数
MAX_WAV_COUNT = 3000


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
        with open(file_path, "w", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            writer.writerows(list)

    # 起こりそうな例外をキャッチ
    except FileNotFoundError as e:
        print(e)
    except csv.Error as e:
        print(e)


def flatten_with_any_depth(nested_list):

    """深さ優先探索の要領で入れ子のリストをフラットにする関数"""
    # フラットなリストとフリンジを用意
    flat_list = []
    fringe = [nested_list]

    while len(fringe) > 0:
        node = fringe.pop(0)
        # ノードがリストであれば子要素をフリンジに追加
        # リストでなければそのままフラットリストに追加
        if isinstance(node, list):
            fringe = node + fringe
        else:
            flat_list.append(node)

    return flat_list


# フォルダ内の全ファイル消し
def clear_dir(dir_path):

    os.chdir(dir_path)
    files = glob.glob("*.*")

    for filename in files:
        os.unlink(filename)
        

def make_resource(word, wav_folder_paths):

    i = 0
    source_elements = []

    for wav_folder_path in wav_folder_paths:

        count = 1
        wav_file_paths = glob.glob("%s/*.wav" % wav_folder_path)

        for wav_file_path in wav_file_paths:

            base = os.path.basename(wav_file_path)
            name, ext = os.path.splitext(base)
            trans_name = name.replace(".", "-").replace(" ", "_")

            # 先頭に0を付ける工夫
            data_size = int(math.log10(MAX_WAV_COUNT) + 1)
            count_size = int(math.log10(count) + 1)
            new_count = "0" * (data_size - count_size) + str(count)

            new_name = "%s/segment_%s_%s_%s" % (TEMP_DIR_PATH, WAV_LABELS[i], new_count, trans_name)
            source_elements.append([new_name, CLASS_LABELS[i], wav_file_path])

            word_file = codecs.open("%s.txt" % new_name, "w", "utf-8")
            word_file.write(word + "\n")
            word_file.close

            shutil.copyfile(wav_file_path, "%s.wav" % new_name)

            count += 1
        i += 1

    return source_elements


def execute_segmentation():

    try:
        subprocess.run(SEGMENTATION_COMMAND, shell=True, check=True)
    except subprocess.CalledProcessError:
        print("外部プログラムの実行に失敗しました", file=sys.stderr)


def pick_score(file_path, word_length):

    labs = read_csv(file_path=file_path, delimiter=" ")
    segments = []

    if len(labs) == 0:
        segments += [0] * word_length

    else:
        # 前後の消音のセグメントは抜かす
        for i in range(1, len(labs) - 1):

            # 特徴量とするセグメントの選抜
            segments.append(labs[i][2])

    return segments


def pick_frame(file_path, word_length):

    labs = read_csv(file_path=file_path, delimiter=" ")
    segments = []

    if len(labs) == 0:
        segments = [0] * word_length * 2
    else:
        # 前後の消音のセグメントは抜かす
        for i in range(1, len(labs) - 1):

            # 特徴量とするセグメントの選抜
            segments.append(labs[i][0])
            segments.append(labs[i][1])

    return segments


def load_wave_librosa(file_path):

    # wavファイルを読み込む (ネイティブサンプリングレートを使用)
    wave, sr = librosa.load(file_path, sr=None)
    return wave, sr


# 高域強調
def preEmphasis(wave, p=0.97):

    # 係数 (1.0, -p) のFIRフィルタを作成
    return scipy.signal.lfilter([1.0, -p], 1, wave)


def get_mfcc_librosa(file_path):

    wave, sr = load_wave_librosa(file_path)

    p = 0.97
    wave = preEmphasis(wave, p)

    n_fft = 512 # ウィンドウサイズ
    hop_length_sec = 0.010 # ずらし幅（juliusの解像度に合わせる）
    n_mfcc = 13 # mfccの次元数

    # mfccを求める
    mfccs = librosa.feature.mfcc(wave, n_fft=n_fft, hop_length=librosa.time_to_samples(hop_length_sec, sr),
                                 sr=sr, n_mfcc=n_mfcc)

    # mfccの1次元はいらないから消す
    mfccs = np.delete(mfccs, 0, axis=0)

    # 対数パワー項を末尾に追加
    S = cal_logpower_librosa(wave, sr)
    mfccs = np.vstack((mfccs, S))

    # mfcc delta* を計算する
    delta_mfccs  = librosa.feature.delta(mfccs)  # mfcc delta

    delta2_mfccs = librosa.feature.delta(mfccs, order=2)  # mfcc delta2

    all_mfccs = mfccs.tolist() + delta_mfccs.tolist() + delta2_mfccs.tolist()

    return all_mfccs


# 対数パワースペクトルの計算
def cal_logpower_librosa(wave, sr):

    n_fft = 512 # ウィンドウサイズ
    hop_length_sec = 0.010 # ずらし幅（juliusの解像度に合わせる）

    S = librosa.feature.melspectrogram(wave, sr=sr, n_fft=n_fft, hop_length=librosa.time_to_samples(hop_length_sec, sr))
    S = sum(S)
    PS = librosa.power_to_db(S)

    return PS


def get_mfcc_aubio(file_path):

    _p = 0.97
    samplerate = 16000

    # for Computing a Spectrum
    win_s = 512 # Window Size
    hop_size = 160 # Hop Size => シフト幅（0.01[s]にずらすサンプル数（juliusの解像度に合わせる）

    n_filters = 40 # must be 40 for mfcc
    n_coeffs = 13

    src = source(file_path, samplerate, hop_size)
    samplerate = src.samplerate
    total_frames = 0
    total_samples = np.array([])

    pv = pvoc(win_s, hop_size)
    f = filterbank(n_coeffs, win_s)
    f.set_mel_coeffs_slaney(samplerate)
    energies = np.zeros((n_coeffs,))

    while True:
        hop_samples, read = src()  # read hop_size new samples from source
        total_samples = np.append(total_samples, hop_samples)

        fftgrain = pv(hop_samples)
        new_energies = f(fftgrain)
        energies = np.vstack([energies, new_energies])

        total_frames += read   # increment total number of frames
        if read < hop_size:    # end of file reached
            break

    # preEmphasis
    total_samples = preEmphasis(total_samples, _p).astype("float32")

    p = pvoc(win_s, hop_size)
    m = mfcc(win_s, n_filters, n_coeffs, samplerate)
    mfccs = np.zeros([n_coeffs,])
    index = 1

    while True:
        old_frame = hop_size * (index - 1)
        cur_frame = hop_size * index

        if total_frames - old_frame < hop_size:
            samples = total_samples[old_frame:total_frames]

            # ケツを0で埋めてhopサイズに間に合わせる
            samples = np.pad(samples, [0, hop_size - (total_frames - old_frame)], "constant")
        else:
            samples = total_samples[old_frame:cur_frame]

        spec = p(samples)
        mfcc_out = m(spec)
        mfccs = np.vstack((mfccs, mfcc_out))

        if total_frames - old_frame < hop_size:
            break
        index += 1

    # mfccの1次元はいらないから消す
    mfccs = np.delete(mfccs, 0, axis=1)

    energies = np.mean(energies, axis=1)

    # 対数パワー項を末尾に追加
    mfccs = np.hstack((mfccs, energies.reshape(energies.shape[0], 1)))

    deltas = np.diff(mfccs, axis=0)
    deltas = np.pad(deltas, [(1, 0), (0, 0)], "constant")

    ddeltas = np.diff(deltas, axis=0)
    ddeltas = np.pad(ddeltas, [(1, 0), (0, 0)], "constant")

    mfccs = mfccs.transpose()
    deltas = deltas.transpose()
    ddeltas = ddeltas.transpose()

    all_mfccs = mfccs.tolist() + deltas.tolist() + ddeltas.tolist()

    print("Get MFCC in " + file_path + " ...")

    return all_mfccs


def section_average(input_paths, division_number):

    if len(input_paths) < division_number:

        print("０埋めで音の長さを規定値に伸長...")

        additions = [0.0] * (division_number - len(input_paths))
        input_paths += additions

    size = int(len(input_paths) // division_number)
    mod = int(len(input_paths) % division_number)

    index_list = [size] * division_number
    if mod != 0:
        for i in range(mod):
            index_list[i] += 1

    section_averages = []
    i = 0

    for index in index_list:
        section_averages.append(mean(input_paths[i: i + index]))
        i += index

    return section_averages


if __name__ == "__main__":

    if not os.path.isdir(TEMP_DIR_PATH):
        os.makedirs(TEMP_DIR_PATH)

    clear_dir(TEMP_DIR_PATH)

    setting = read_csv(file_path=SETTING_FILE_PATH, delimiter=",")

    # 訓練とテスト用のデータをまとめて作る
    for method in range(0, METHOD_PATTERN):

        # temp内の全ファイル消し
        clear_dir(TEMP_DIR_PATH)

        if method == 0:
            learning_file_path = TRAINING_FILE_PATH
            wav_folder_paths = TRAINING_WAV_DIR_PATHS
        else:
            learning_file_path = TEST_FILE_PATH
            wav_folder_paths = TEST_WAV_DIR_PATHS

        word = setting[0][1]
        source_elements = make_resource(word, wav_folder_paths)
        print("All resource files copy in temp dir...")

        execute_segmentation()

        word_length = len(setting[0][2])

        trains = []
        for source_element in source_elements:

            frames = pick_frame(source_element[0] + ".lab", word_length)
            print("Pick score in %s.lab..." % source_element[0])

            if GET_MFCC_METHOD == "librosa":
                input_mfccs = get_mfcc_librosa(source_element[2])

            elif GET_MFCC_METHOD == "aubio":
                input_mfccs = get_mfcc_aubio(source_element[2])

            print("Get MFCC & Delta with Power in %s..." % source_element[2])

            frame2ds = []
            for i in range(int(len(frames) / 2)):
                frame2ds.append([frames[i * 2], frames[i * 2 + 1]])

            scores = pick_score(source_element[0] + ".lab", word_length)
            print("Pick frame in %s.lab..." % source_element[0])
            scores = [float(s) for s in scores]

            features = []
            for input_mfcc in input_mfccs:

                section_in_mora_averages = []
                for frame2d in frame2ds:

                    # juliusの解像度は0.01[s]（10[ms]）なのでフレーム数は100をかけて復元
                    start_frame = math.floor(float(frame2d[0]) * 100.0)
                    end_frame = math.floor(float(frame2d[1]) * 100.0)

                    # 1音素を分割しそれぞれの区間平均をとる
                    section_in_mora_averages.append(section_average(input_mfcc[start_frame: end_frame],
                                                                    MORA_TIME_DIVISION_NUMBER))

                features.append(section_in_mora_averages)

            # ケツにセグメントのスコア追加
            features.append(scores)

            features = flatten_with_any_depth(features)
            features.insert(0, source_element[1])

            trains.append(features)

        write_csv(learning_file_path, trains)

    clear_dir(TEMP_DIR_PATH)
    # shutil.rmtree(TEMP_DIR_PATH)

    print("\nAll process completed...")
