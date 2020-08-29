import unittest
from francis import model_adaptor
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal


class ModelAdaptorTest(unittest.TestCase):
    def test_training_testing_split(self):
        mock_df = pd.DataFrame(
            {
                "bird_name": [
                    "CommonBlackbird",
                    "Wren",
                    "Chicken",
                    "Duck",
                    "CommonBlackBird",
                ],
                "mfcc_data": [
                    np.array([1, 2, 3, 4, 5]),
                    np.array([6, 7, 8, 9, 10]),
                    np.array([11, 12, 13, 14, 15]),
                    np.array([16, 17, 18, 19, 20]),
                    np.array([21, 22, 23, 24, 25]),
                ],
            }
        )

        train_in, test_in, train_out, test_out = model_adaptor.call(
            mock_df, test_size=0.2, random_state=42
        )

        self.assertEqual(len(train_in), 4)
        self.assertEqual(len(train_out), 4)
        self.assertEqual(len(test_in), 1)
        self.assertEqual(len(test_out), 1)

    def test_label_encoding(self):

        mock_df = pd.DataFrame(
            {
                "bird_name": ["CommonBlackbird", "Wren"],
                "mfcc_data": [np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10])],
            }
        )

        train_in, test_in, train_out, test_out = model_adaptor.call(
            mock_df, test_size=0.5, random_state=42
        )

        assert_array_equal(train_in[0], [1, 0])
        assert_array_equal(train_out[0], np.array([1, 2, 3, 4, 5]))
