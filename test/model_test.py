import unittest
from francis import model
import numpy as np
import librosa


def noisy_audio(seconds):
    return np.random.uniform(low=-1.0, high=1.0, size=(seconds * 22050,))


def make_spectrogram(seconds):
    return np.reshape(
        librosa.power_to_db(
            librosa.feature.melspectrogram(noisy_audio(seconds), sr=22050)
        ),
        (128, 216, 1),
    )


class ModelTest(unittest.TestCase):
    def test_model_make(self):

        model.make(2)

    def test_model_test(self):
        mock_input = np.asarray([make_spectrogram(5), make_spectrogram(5)])
        mock_output = np.array([[0.0, 1.0], [1.0, 0.0]])

        the_model = model.make(len(mock_input))

        model.test(the_model, mock_input, mock_output)

    def train_model_test(self):
        mock_input = np.asarray([make_spectrogram(5), make_spectrogram(5)])
        mock_output = np.array([[0.0, 1.0], [1.0, 0.0]])

        the_model = model.make(len(mock_input))

        model.train(the_model, mock_input, mock_output, batch_size=1, epochs=5)


rand = ModelTest()
rand.test_model_test()
