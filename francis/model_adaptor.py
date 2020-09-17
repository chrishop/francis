from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from math import sqrt


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


def _prediction_filter(predictions, categories, cuttoff_fun=lambda x: 0):
    # function returns index of most likely bird or [last index + 1] if no bird exceeds the cutoff
    # default cuttoff_fun sets cuttoff always to 0, meaning that the max bird will always be chosen
    cutoff = cuttoff_fun(len(categories))
    max_arg = np.argmax(predictions, axis=1)
    max_value = np.amax(predictions, axis=1)
    filtered_prediction = np.where(max_value < cutoff, len(categories), max_arg)
    return filtered_prediction


def predicted_bird_timestamp(predictions, categories, seconds=5):
    # condenses bird predictions to a time stamp for predicted birds only
    predicted_bird_index = _prediction_filter(
        predictions, categories, cuttoff_fun=lambda x: 1 / sqrt(x)
    )
    bird_prediction = [[] for i in range(len(categories) + 1)]
    for i, prediction in enumerate(predicted_bird_index):
        bird_prediction[prediction].append(i * seconds)
    return bird_prediction


def prediction_string_format(bird_prediction, categories):
    output = ""
    categories.append("Unidentified audio")
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
