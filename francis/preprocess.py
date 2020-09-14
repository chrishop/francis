# takes a non processed audio file and returns it when it has been
# normalised (librosa already does this) , had a high pass filter and noise reduction applied.
# high pass filter should only be applied to blackbird audio files not the others
from francis import noise_reduction
from francis import high_pass_filter
from progress.bar import Bar


def process(df, sample_rate, pre_process=True):
    if pre_process:
        bar2 = Bar("noise reducing and high pass filtering", max=len(df))
        for i, row in df.iterrows():
            bar2.next()
            row["audio_buffer"] = noise_reduction.process(
                (row["audio_buffer"], sample_rate)
            )
            row["audio_buffer"] = high_pass_filter.process(
                (row["audio_buffer"], sample_rate)
            )
        bar2.finish()
        print()
    return df
