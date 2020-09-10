from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np


def adapt(the_dataframe, test_size=0.2, random_state=42, split_override=False):
    birdnames = the_dataframe["label"].to_numpy()
    spectrograms = __shape_spectrograms(the_dataframe["spectrogram"].to_numpy())

    birdnames_binary = __label(birdnames)

    return __split(birdnames_binary, spectrograms, test_size, random_state)


def adapt_spectrograms(the_dataframe):
    return np.asarray(__shape_spectrograms(the_dataframe["spectrogram"].to_numpy()))


def adapt_predictions(predictions, categories):
    # convert predictions to integer list
    predictions_as_num = np.argmax(predictions, axis=1)

    # uses categories array to translate number to category name
    return list(map(lambda x: categories[x], predictions_as_num))


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
