# this is where or the neural network logic will go once we have it
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


def make(input_data, expected_output):
    model = Sequential()
    model.add(
        Conv2D(
            filters=16,
            kernel_size=2,
            input_shape=(128, 216, 1),
            activation="relu",
        )
    )

    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(filters=32, kernel_size=2, activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2), activation="softmax")

    model.compile(
        loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam"
    )

    model.summary()

    return model


def test(model, input_data, expected_data, verbose=0):
    return model.evaluate(input_data, expected_data, verbose=verbose)


def train(model, input_data, expected_data, batch_size, epochs, verbose=0):
    return model.fit(
        input_data,
        expected_data,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
    )
