# here we split the files into 5s segments
# then we calculate the standard deviation of each
# if its below a cut off point then its removed from
# the dataset.
# onwards from here the data is sent to be made into a spectrogram

import pandas as pd
import numpy as np
from progress.bar import Bar


def call(df):
    return __split(df)


def __split(pre_df):
    labeled = []
    bar = Bar("splitting audio into 5 second chunks", max=len(pre_df))
    for i, row in pre_df.iterrows():
        bar = Bar("splitting audio into 5 second chunks", max=len(pre_df))
        try:
            split_buffer = __split_buffer(row["audio_buffer"], 22050, 5)
            filtered_buffer = __filter_chunks(
                split_buffer, type="quartile", cutoff="default"
            )
            for buffer in filtered_buffer:
                labeled.append((row["label"], buffer))

            bar.next()
        except ZeroDivisionError:
            print("oops that audio was less than 5s")

    bar.finish()

    return pd.DataFrame(labeled, columns=["label", "audio_buffer"])


def __split_buffer(buffer, sample_rate, seconds):
    chunk_size = sample_rate * seconds
    chunk_number = int(np.shape(buffer)[0] // chunk_size)
    cutoff = int(chunk_size * chunk_number)

    return np.split(buffer[:cutoff], chunk_number)


def __filter_chunks(split_audio, type="quartile", cutoff="default"):
    """type="quartile" uses the 95% quartile of each chunk as its filter criterea,
    type="std" uses the standand deviation of each chunk as its filter criterea
    """
    abs_audio = np.array(
        [np.absolute(chunk) for chunk in split_audio]
    )  # converts amplitudes to positive to be able to perform quartile
    if type == "quartile":
        if cutoff == "default":
            cutoff = 0.15  # 15% default cuttoff for 95% quartile
        chunk_measure = [
            np.percentile(chunk, 95) for chunk in abs_audio
        ]  # calcuate 95% quartile of each chunk
    elif type == "std":
        if cutoff == "default":
            cutoff = 0.2  # 20% default cuttoff for std
        chunk_measure = [
            np.ndarray.std(chunk) for chunk in abs_audio
        ]  # calcuate std of each chunk

    max_chunk_measure = max(
        chunk_measure
    )  # calculate largest 95% quartile or std out of all the chunks
    chunk_mask = [
        chunk > cutoff * max_chunk_measure for chunk in chunk_measure
    ]  # create a mask based on the max 95% or std and the cutoff
    # print(
    #    f"{len(chunk_mask) - sum(chunk_mask)} chunks out of {len(chunk_mask)} chunks in the audio where filtered out"
    # )
    filtered_audio = [
        chunk for i, chunk in enumerate(split_audio) if chunk_mask[i]
    ]  # masks out the chunks which are bellow the cuttoff
    return filtered_audio
