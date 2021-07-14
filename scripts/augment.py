#!/usr/bin/env python
# coding: utf-8

import os
import glob
import shutil
import subprocess


# Path
BASE_ABSOLUTE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/../"
TEMP_DIR_PATH = BASE_ABSOLUTE_PATH + "temp"
DATA_DIR_PATH = BASE_ABSOLUTE_PATH + "data"
TRAINING_WAV_DIR_PATHS = [DATA_DIR_PATH + "/wavs/bad_wav", DATA_DIR_PATH + "/wavs/ok_wav"]
OUTPUT_WAV_DIR_PATHS = [DATA_DIR_PATH + "/wavs/augment/bad_wav", DATA_DIR_PATH + "/wavs/augment/ok_wav"]

CHANNELS = 1
RATE = 16000
CHUNK = 1024

# MODFY_PITCH_RATIOS = [0.25, 0.5, 0.75, 1.25, 1.5, 1.75]

IS_CHILD_VOICE = True
IS_MODFY_PITCH = False


def extract_pitch(raw_file, pitch_file):

    """ピッチパラメータの抽出"""
    cmd = "..\\SPTK-3.11\\bin\\x2x +sf %s | ..\\SPTK-3.11\\bin\\pitch -a 1 -s 16 -p 80 > %s" % (raw_file, pitch_file)
    subprocess.call(cmd, shell=True)


def extract_mcep(raw_file, mcep_file):

    """メルケプストラムパラメータの抽出"""
    cmd = "..\\SPTK-3.11\\bin\\x2x +sf %s | ..\\SPTK-3.11\\bin\\frame -p 80 | ..\\SPTK-3.11\\bin\\window | ..\\SPTK-3.11\\bin\\mcep -m 25 -a 0.42 > %s" % (raw_file, mcep_file)
    subprocess.call(cmd, shell=True)


def modify_pitch(m, pitch_file, mcep_file, raw_file):

    """ピッチを変形して再合成
    mが1より大きい => 低い声
    mが1より小さい => 高い声"""
    cmd = "..\\SPTK-3.11\\bin\\sopr -m %f %s | ..\\SPTK-3.11\\bin\\excite -p 80 | ..\\SPTK-3.11\\bin\\mlsadf -m 25 -a 0.42 -p 80 %s | ..\\SPTK-3.11\\bin\\clip -y -32000 32000 | ..\\SPTK-3.11\\bin\\x2x +fs > %s" % (m, pitch_file, mcep_file, raw_file)
    subprocess.call(cmd, shell=True)


def modify_speed(frame_shift, pitch_file, mcep_file, raw_file):

    """話速を変形して再合成
    frame_shiftが小さい => 早口
    frame_shiftが大きい => ゆっくり"""
    cmd = "..\\SPTK-3.11\\bin\\excite -p %f %s | ..\\SPTK-3.11\\bin\\mlsadf -m 25 -a 0.42 -p %f %s | ..\\SPTK-3.11\\bin\\clip -y -32000 32000 | ..\\SPTK-3.11\\bin\\x2x +fs > %s" % (frame_shift, pitch_file, frame_shift, mcep_file, raw_file)
    subprocess.call(cmd, shell=True)


def hoarse_voice(pitch_file, mcep_file, raw_file):

    """ささやき声"""
    modify_pitch(0, pitch_file, mcep_file, raw_file)


def robot_voice(frame_period, record_seconds, mcep_file, raw_file):

    """ロボット声
    frame_periodが小さい => 低い
    frame_periodが大きい => 高い"""
    sequence_length = record_seconds * RATE * frame_period
    cmd = "..\\SPTK-3.11\\bin\\train -p %d -l %d | ..\\SPTK-3.11\\bin\\mlsadf -m 25 -a 0.42 -p 80 %s | ..\\SPTK-3.11\\bin\\clip -y -32000 32000 | ..\\SPTK-3.11\\bin\\x2x +fs > %s" % (frame_period, sequence_length, mcep_file, raw_file)
    subprocess.call(cmd, shell=True)


def child_voice(pitch_file, mcep_file, raw_file):

    """子供声"""
    cmd = "..\\SPTK-3.11\\bin\\sopr -m 0.4 %s | ..\\SPTK-3.11\\bin\\excite -p 80 | ..\\SPTK-3.11\\bin\\mlsadf -m 25 -a 0.1 -p 80 %s | ..\\SPTK-3.11\\bin\\clip -y -32000 32000 | ..\\SPTK-3.11\\bin\\x2x +fs > %s" % (pitch_file, mcep_file, raw_file)
    subprocess.call(cmd, shell=True)


def deep_voice(pitch_file, mcep_file, raw_file):

    """太い声"""
    cmd = "..\\SPTK-3.11\\bin\\sopr -m 2.0 %s | ..\\SPTK-3.11\\bin\\excite -p 80 | ..\\SPTK-3.11\\bin\\mlsadf -m 25 -a 0.6 -p 80 %s | ..\\SPTK-3.11\\bin\\clip -y -32000 32000 | ..\\SPTK-3.11\\bin\\x2x +fs > %s" % (pitch_file, mcep_file, raw_file)
    subprocess.call(cmd, shell=True)


def raw2wav(raw_file, wav_file):

    cmd = "..\\sox-14.4.2\\sox -e signed-integer -c %d -b 16 -r %d %s %s" % (CHANNELS, RATE, raw_file, wav_file)
    subprocess.call(cmd, shell=True)


def wav2raw(wav_file, raw_file):

    cmd = "..\\sox-14.4.2\\sox %s %s" % (wav_file, raw_file)
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":

    for output_wav_dir_path in OUTPUT_WAV_DIR_PATHS:
        if os.path.isdir(output_wav_dir_path):
            shutil.rmtree(output_wav_dir_path)
        os.makedirs(output_wav_dir_path)

    if os.path.isdir(TEMP_DIR_PATH):
        shutil.rmtree(TEMP_DIR_PATH)
    os.makedirs(TEMP_DIR_PATH)

    print("Training Data Augment Start...")

    for index in range(len(TRAINING_WAV_DIR_PATHS)):

        wav_file_paths = glob.glob("%s/*.wav" % TRAINING_WAV_DIR_PATHS[index])

        output_dir_path = OUTPUT_WAV_DIR_PATHS[index]

        for wav_file_path in wav_file_paths:

            if not os.path.isdir(output_dir_path):
                os.mkdir(output_dir_path)

            pitch_file = BASE_ABSOLUTE_PATH + "temp/%s.pitch" % os.path.basename(wav_file_path)
            mcep_file = BASE_ABSOLUTE_PATH + "temp/%s.mcep" % os.path.basename(wav_file_path)
            input_raw_file = BASE_ABSOLUTE_PATH + "temp/%s_input.raw" % os.path.basename(wav_file_path)

            print("*** Now converting %s wav to raw ..." % os.path.basename(wav_file_path))
            wav2raw(wav_file_path, input_raw_file)

            # パラメータ抽出
            print("*** extract pitch %s..." % os.path.basename(wav_file_path))
            extract_pitch(input_raw_file, pitch_file)

            print("*** extract mel cepstrum %s..." % os.path.basename(wav_file_path))
            extract_mcep(input_raw_file, mcep_file)

            # パラメータ変形いろいろ
            print("*** modify parameters %s..." % os.path.basename(wav_file_path))

            # どれか一つしか有効にできない

            '''
            if IS_MODFY_PITCH:

                for pitch_ratio in MODFY_PITCH_RATIOS:

                    output_raw_file = BASE_ABSOLUTE_PATH + "temp/%s_output_ratio%f.raw" % (os.path.basename(wav_file_path), pitch_ratio)
                    modify_pitch(pitch_ratio, pitch_file, mcep_file, output_raw_file)

                    save_file_path = output_dir_path + "/modify_pitch_ratio%f_%s" % (pitch_ratio, os.path.basename(wav_file_path))
                    raw2wav(output_raw_file, save_file_path)
            '''

            if IS_CHILD_VOICE:

                output_raw_file = BASE_ABSOLUTE_PATH + "temp/%s_child_voice.raw" % os.path.basename(wav_file_path)

                print("*** voice change to child voice %s..." % os.path.basename(wav_file_path))
                child_voice(pitch_file, mcep_file, output_raw_file)

                save_file_path = output_dir_path + "/child_voice_%s" % os.path.basename(wav_file_path)

                print("*** Now converting %s raw to wav ..." % os.path.basename(wav_file_path))
                raw2wav(output_raw_file, save_file_path)

            # modify_pitch(0.3, pitch_file, mcep_file, output_file)
            # modify_speed(300, pitch_file, mcep_file, output_file)
            # hoarse_voice(pitch_file, mcep_file, output_file)
            # robot_voice(100, record_seconds, mcep_file, output_file)
            # child_voice(pitch_file, mcep_file, output_file)
            # deep_voice(pitch_file, mcep_file, output_file)

            # save_file_path = output_dir_path + "/deep_voice_" + os.path.basename(wav_file_path)

    if os.path.isdir(TEMP_DIR_PATH):
        shutil.rmtree(TEMP_DIR_PATH)

    print("\nAll process completed...")
