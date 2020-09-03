import librosa
import glob
import pandas as pd
import os
import xenocanto
from pydub import AudioSegment

# loads all files in folders and subfolders
# into dataframe as an audio buffer and a sample rate


def download(xeno_canto_args):
    xenocanto.download(xeno_canto_args)
    return convert_to_wav(os.getcwd() + "/dataset/audio")


def load_into_df(filepaths):
    file_data = []
    for i, path in enumerate(filepaths):
        print(f"loading into dataframe: {i + 1}/{len(filepaths)}")
        audio_buffer, _ = librosa.load(path)
        the_id = __filename(path)
        label = __foldername(path)
        file_data.append((the_id, label, audio_buffer))

    return pd.DataFrame(file_data, columns=["id", "label", "audio_buffer"])


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


def __wav_path(filepath):
    return filepath.split(".")[0] + ".wav"


def __filename(filepath):
    return filepath.split("/")[-1].split(".")[0]


def __foldername(filepath):
    return filepath.split("/")[-2]
