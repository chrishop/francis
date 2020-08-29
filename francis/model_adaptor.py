from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def call(the_dataframe, test_size=0.2, random_state=42):
    birdnames = the_dataframe["bird_name"].to_numpy()
    mfcc_data = the_dataframe["mfcc_data"].to_numpy()

    birdnames_binary = __label(birdnames)

    return __split(birdnames_binary, mfcc_data, test_size, random_state)


def __label(birdnames):
    """
    compresses the label data
    so its easier for the nn model to handle
    """
    return to_categorical(LabelEncoder().fit_transform(birdnames))


def __split(birdnames_binary, mfcc_data, test_size, random_state):
    """
    returns 4 lists
        training input,
        test input,
        training expected output,
        test expected output
    """
    return train_test_split(
        birdnames_binary, mfcc_data, test_size=test_size, random_state=random_state
    )
