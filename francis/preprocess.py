# takes a non processed audio file and returns it when it has been
# normalised (librosa already does this) , had a high pass filter and noise reduction applied.
# high pass filter should only be applied to blackbird audio files not the others
from francis import split_filter
from francis import noise_reduction
from francis import high_pass_filter


def process(df):
    for i, row in df.iterrows():
        # need to include 22050 because of the way nose_reduction and high_pass_filter process audiodata
        row["audio_buffer"] = noise_reduction.process((row["audio_buffer"], 22050))
        row["audio_buffer"] = high_pass_filter.process((row["audio_buffer"], 22050))
    df = split_filter.call(df)
    return df
