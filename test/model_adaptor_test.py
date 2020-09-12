import unittest
from francis import model_adaptor
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
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


class ModelAdaptorTest(unittest.TestCase):
    def test_training_testing_split(self):

        a_spectrogram = make_spectrogram(5)

        mock_df = pd.DataFrame(
            {
                "label": [
                    "CommonBlackbird",
                    "Wren",
                    "Chicken",
                    "Duck",
                    "CommonBlackBird",
                ],
                "spectrogram": [
                    a_spectrogram,
                    a_spectrogram,
                    a_spectrogram,
                    a_spectrogram,
                    a_spectrogram,
                ],
            }
        )

        train_out, test_out, train_in, test_in = model_adaptor.adapt(
            mock_df, test_size=0.2, random_state=42
        )

        self.assertEqual(len(train_in), 4)
        self.assertEqual(len(train_out), 4)
        self.assertEqual(len(test_in), 1)
        self.assertEqual(len(test_out), 1)

    def test_label_encoding(self):

        blackbird_spectrogram = make_spectrogram(5)
        duck_spectrogram = make_spectrogram(5)

        mock_df = pd.DataFrame(
            {
                "label": ["CommonBlackbird", "Wren"],
                "spectrogram": [blackbird_spectrogram, duck_spectrogram],
            }
        )

        train_out, _, _, _ = model_adaptor.adapt(
            mock_df, test_size=0.5, random_state=42
        )

        assert_array_equal(train_out[0], [1, 0])

    def test_list_is_correct_format_for_model(self):
        # to be accepted by keras, the data needs to be in a pure numpy array

        blackbird_spectrogram = make_spectrogram(5)
        duck_spectrogram = make_spectrogram(5)

        mock_df = pd.DataFrame(
            {
                "label": ["CommonBlackbird", "Duck", "Sparrow", "tit"],
                "spectrogram": [
                    blackbird_spectrogram,
                    duck_spectrogram,
                    blackbird_spectrogram,
                    duck_spectrogram,
                ],
            }
        )

        train_out, test_out, train_in, test_in = model_adaptor.adapt(
            mock_df, test_size=0.5, random_state=42
        )

        # the top list must be a numpy array
        self.assertTrue(type(train_in) is np.ndarray)
        self.assertTrue(type(test_in) is np.ndarray)
        # each item must have this shape
        self.assertEqual(np.shape(test_in[0]), (128, 216, 1))

    def test_no_test_training_split_with_argument_override(self):
        blackbird_spectrogram = make_spectrogram(5)
        wren_spectrogram = make_spectrogram(5)

        mock_df = pd.DataFrame(
            {
                "label": ["CommonBlackbird", "Wren"],
                "spectrogram": [blackbird_spectrogram, wren_spectrogram],
            }
        )

        results_in = model_adaptor.adapt_spectrograms(mock_df)

        # the output is an array with hot encoding of categories
        self.assertEqual(results_in.shape, (2, 128, 216, 1))

    def test_adapt_prediction(self):
        categories = ["CommonBlackbird", "EurasianRobin", "Wren"]
        predictions = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]

        expected = ["CommonBlackbird", "Wren", "EurasianRobin"]

        result = model_adaptor.adapt_predictions(predictions, categories)

        self.assertEqual(expected, result)
