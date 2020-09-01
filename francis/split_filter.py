# here we split the files into 5s segments
# then we calculate the standard deviation of each
# if its below a cut off point then its removed from
# the dataset.
# onwards from here the data is sent to be made into a spectrogram

import pandas as pd
import numpy as np


def call(df):
    return __split(df)


def __split(pre_df):
    labeled = []
    for i, row in pre_df.iterrows():
        split_buffer = __split_buffer(row["audio_buffer"], 22050, 5)
        for buffer in split_buffer:
            labeled.append((row["label"], buffer))

    return pd.DataFrame(labeled, columns=["label", "audio_buffer"])


def __split_buffer(buffer, sample_rate, seconds):
    chunk_size = sample_rate * seconds
    chunk_number = np.shape(buffer)[0] // chunk_size
    cutoff = chunk_size * chunk_number

    return np.split(buffer[:cutoff], chunk_number)
