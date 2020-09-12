import librosa
import glob
import pandas as pd
import os
import xenocanto
from pydub import AudioSegment
import numpy as np
import json
import math
import h5py

# loads all files in folders and subfolders
# into dataframe as an audio buffer and a sample rate


def download(xeno_canto_args, delete_old=False):
    xenocanto.download(xeno_canto_args)
    return convert_to_wav(os.getcwd() + "/dataset/audio", delete_old=delete_old)


def load_into_df(folderpath):
    filepaths = glob.glob(folderpath + "/**/*.wav", recursive=True)
    file_data = []
    for i, path in enumerate(filepaths):
        print(f"loading into dataframe: {i + 1}/{len(filepaths)}")
        file_data.append(__get_file_data(path))

    return pd.DataFrame(file_data, columns=["id", "label", "audio_buffer"])


def load_file_into_df(filepath: str):
    return pd.DataFrame(
        [__get_file_data(filepath)], columns=["id", "label", "audio_buffer"]
    )


def save_df(folderpath: str, df, rows_per_file=1000):
    rows = len(df.index)
    number_of_files = math.ceil(rows / rows_per_file)
    split_df = np.array_split(df, number_of_files)

    files_made = []
    for i, mini_df in enumerate(split_df):
        filepath = folderpath + f"/train_test_data_{i}.parquet"

        try:
            mini_df.to_parquet(filepath)
            files_made.append(filepath)
        except Exception as e:
            print(e)
            print("oops couldn't handle that one")

    return files_made


def load_df(folderpath: str):
    return pd.read_parquet(folderpath)


def save_categories(filepath: str, df):
    categories_list = np.string_(df["label"].unique().tolist())

    with h5py.File(filepath, "w") as hf:
        hf.create_dataset("categories", data=categories_list)

    return categories_list


def load_categories(filepath):
    with h5py.File(filepath, "r") as hf:
        return __list_to_utf8(np.array(hf["categories"]).tolist())


def convert_to_wav(folderpath, delete_old=False):
    filepaths = glob.glob(folderpath + "/**/*.mp3", recursive=True)
    converted = []
    for i, path in enumerate(filepaths):
        print(
            f"converting {__filename(path)}.mp3 to wav file: {i + 1}/{len(filepaths)}"
        )

        # reads mp3 and writes wav
        AudioSegment.from_mp3(path).export(__wav_path(path), format="wav").close()

        if delete_old:
            print(f"removing {__filename(path)}.mp3")
            os.remove(path)

        converted.append(__wav_path(path))

    return converted


def __get_file_data(path: str) -> tuple:
    the_id = __filename(path)
    label = __foldername(path)
    audio_buffer, _ = librosa.load(path)

    return (the_id, label, audio_buffer)


def __wav_path(filepath):
    return filepath.split(".")[0] + ".wav"


def __filename(filepath):
    return filepath.split("/")[-1].split(".")[0]


def __foldername(filepath):
    return filepath.split("/")[-2]


def __list_to_utf8(string_list: list) -> list:
    return [string.decode("utf8") for string in string_list]
