# takes a non processed audio file and returns it when it has been
# normalised (librosa already does this) , had a high pass filter and noise reduction applied.
# high pass filter should only be applied to blackbird audio files not the others
from francis import split_filter
from francis import noise_reduction
from francis import high_pass_filter
from progress.bar import Bar


def process(df):
    bar2 = Bar("noise reducing and high pass filtering", max=len(df))
    for i, row in df.iterrows():
        bar2.next()
        row["audio_buffer"] = noise_reduction.process((row["audio_buffer"], 22050))
        row["audio_buffer"] = high_pass_filter.process((row["audio_buffer"], 22050))
    bar2.finish
    print()
    df = split_filter.call(df)
    return df
