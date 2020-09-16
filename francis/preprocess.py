# takes a non processed audio file and returns it when it has been
# normalised (librosa already does this) , had a high pass filter and noise reduction applied.
# high pass filter should only be applied to blackbird audio files not the others
from francis import noise_reduction
from francis import high_pass_filter


def process(df, sample_rate=22050, pre_process=True, bar_config=None):
    if pre_process:
        for i, row in df.iterrows():
            if bar_config:
                bar_config.next()
            row["audio_buffer"] = noise_reduction.process(
                (row["audio_buffer"], sample_rate)
            )
            row["audio_buffer"] = high_pass_filter.process(
                (row["audio_buffer"], sample_rate)
            )
        if bar_config:
            bar_config.finish()
    return df
