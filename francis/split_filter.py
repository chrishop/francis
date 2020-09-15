# here we split the files into 5s segments
# then we calculate the standard deviation of each
# if its below a cut off point then its removed from
# the dataset.
# onwards from here the data is sent to be made into a spectrogram

from pandas import DataFrame
import numpy as np
from francis.output_progress import default_bar


def call(
    df,
    sample_rate=22050,
    second_split=5,
    split_type="quartile",
    split_cutoff=0.15,
):
    return __split(df, sample_rate, second_split, split_type, split_cutoff)


def __split(
    pre_df, sample_rate, second_split, split_type, split_cutoff, bar_config=default_bar
):
    labeled = []
    bar = bar_config("chunking and filtering audio... \t\t\t", len(pre_df))
    for i, row in pre_df.iterrows():
        try:
            split_buffer = __split_buffer(
                row["audio_buffer"], sample_rate, second_split
            )
            filtered_buffer = __filter_chunks(split_buffer, split_type, split_cutoff)
            for buffer in filtered_buffer:
                labeled.append((row["label"], buffer))

            bar.next()
        except ZeroDivisionError:
            pass

    bar.finish()
    return DataFrame(labeled, columns=["label", "audio_buffer"])


def __split_buffer(buffer, sample_rate, seconds):
    chunk_size = sample_rate * seconds
    chunk_number = int(np.shape(buffer)[0] // chunk_size)
    cutoff = int(chunk_size * chunk_number)

    return np.split(buffer[:cutoff], chunk_number)


def __filter_chunks(split_audio, split_type, split_cutoff):
    """split_type="quartile" uses the 95% quartile of each chunk as its filter criterea,
    split_type="std" uses the standand deviation of each chunk as its filter criterea
    """
    abs_audio = np.array(
        [np.absolute(chunk) for chunk in split_audio]
    )  # converts amplitudes to positive to be able to perform quartile
    if split_type == "quartile":
        chunk_measure = [
            np.percentile(chunk, 95) for chunk in abs_audio
        ]  # calcuate 95% quartile of each chunk
    elif split_type == "std":
        chunk_measure = [
            np.ndarray.std(chunk) for chunk in abs_audio
        ]  # calcuate std of each chunk

    max_chunk_measure = max(
        chunk_measure
    )  # calculate largest 95% quartile or std out of all the chunks
    chunk_mask = [chunk > split_cutoff * max_chunk_measure for chunk in chunk_measure]
    filtered_audio = [
        chunk for i, chunk in enumerate(split_audio) if chunk_mask[i]
    ]  # masks out the chunks which are bellow the cuttoff
    return filtered_audio
