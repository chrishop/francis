# this is where or the neural network logic will go once we have it
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


def make(num_birds):
    model = Sequential()
    model.add(
        Conv2D(
            filters=16, kernel_size=(3, 3), input_shape=(128, 216, 1), activation="relu"
        )
    )

    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_birds, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam"
    )

    # model.summary()

    return model


def test(model, input_data, expected_data, verbose=0):
    return model.evaluate(input_data, expected_data, verbose=verbose, batch_size=1)


def train(model, input_data, expected_data, batch_size, epochs, verbose=0):
    return model.fit(
        input_data, expected_data, batch_size=batch_size, epochs=epochs, verbose=verbose
    )
