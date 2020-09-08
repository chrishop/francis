from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np


def call(the_dataframe, test_size=0.2, random_state=42, split_override=False):
    birdnames = the_dataframe["label"].to_numpy()
    spectrograms = __shape_spectrograms(the_dataframe["spectrogram"].to_numpy())

    birdnames_binary = __label(birdnames)

    if split_override:
        return birdnames_binary, np.asarray(spectrograms)

    return __split(birdnames_binary, spectrograms, test_size, random_state)


def __label(birdnames):
    """
    compresses the label data
    so its easier for the nn model to handle
    """
    return to_categorical(LabelEncoder().fit_transform(birdnames))


def __split(birdnames_binary, spectrograms, test_size, random_state):
    """
    returns 4 lists
        training input,
        test input,
        training expected output,
        test expected output
    """
    train_out, test_out, train_in, test_in = train_test_split(
        birdnames_binary, spectrograms, test_size=test_size, random_state=random_state
    )

    return train_out, test_out, np.asarray(train_in), np.asarray(test_in)


def __shape_spectrograms(spectrograms):
    shaped_spectrograms = []
    for gram in spectrograms:
        shaped_spectrograms.append(np.reshape(gram, (128, 216, 1)))

    return shaped_spectrograms
