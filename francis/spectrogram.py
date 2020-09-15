#from librosa import power_to_db
import librosa.core
import librosa.feature
from francis.output_progress import default_bar

# this will produce the mfcc image used to train the neural networks


def add_to_df(the_df, bar_config=default_bar):
    bar = bar_config("adding spectrograms \t\t\t\t\t", len(the_df))
    spectrogram_column_data = []
    for i, row in the_df.iterrows():
        spectrogram_column_data.append(__create_spectrogram(row["audio_buffer"]))
        bar.next()
    the_df["spectrogram"] = spectrogram_column_data
    bar.finish()
    return the_df


def __create_spectrogram(audio_buffer):
    return librosa.power_to_db(librosa.feature.melspectrogram(audio_buffer, sr=22050))
