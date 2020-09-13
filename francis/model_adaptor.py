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


def summarize_predictions(predictions, categories, seconds=5):
    # condenses bird predictions to a time stamp for predicted birds only
    predictions_as_num = np.argmax(predictions, axis=1)
    bird_prediction = [[] for i in range(len(categories))]
    output = ""

    for i, prediction in enumerate(predictions_as_num):
        bird_prediction[prediction].append(i * seconds)

    for bird, times in enumerate(bird_prediction):
        if times:
            output += (
                f"{categories[bird]} predicted at times: {str(times)[1:-1]} seconds\n"
            )
    return output[:-1]


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
