import unittest
from francis import spectrogram
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SpectrogramTest(unittest.TestCase):
    def test_spectrogram(self):

        blackbird_song = np.random.uniform(low=-1.0, high=1.0, size=(5 * 22050,))
        sparrow_song = np.random.uniform(low=-1.0, high=1.0, size=(5 * 22050,))

        the_df = pd.DataFrame(
            {
                "label": ["blackbird", "sparrow"],
                "audio_buffer": [blackbird_song, sparrow_song],
            }
        )

        the_df = spectrogram.add_to_df(the_df)

        # there is a new spectrogram column
        self.assertEqual(
            the_df.columns.to_list(), ["label", "audio_buffer", "spectrogram"]
        )

        # the spectrogram has a shape of 128, 216 (shows 5s image)
        self.assertEqual(np.shape(the_df.iloc[0]["spectrogram"]), (128, 216))
