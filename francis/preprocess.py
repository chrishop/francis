# takes a non processed audio file and returns it when it has been
# normalised (librosa already does this) , had a high pass filter and noise reduction applied.
# high pass filter should only be applied to blackbird audio files not the others
from francis import noise_reduction
from francis import high_pass_filter
from francis.output_progress import default_bar

def process(df, sample_rate=22050, pre_process=True, bar_config=default_bar):
    if pre_process:
        bar = bar_config("noise reducing and high pass filtering", len(df))
        for i, row in df.iterrows():
            bar.next()
            row["audio_buffer"] = noise_reduction.process(
                (row["audio_buffer"], sample_rate)
            )
            row["audio_buffer"] = high_pass_filter.process(
                (row["audio_buffer"], sample_rate)
            )
        bar.finish()
        print()
    return df
