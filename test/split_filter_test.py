import unittest
import pandas as pd
import numpy as np
from francis import split_filter
from francis import noise_reduction
import librosa


class SplitFilterTest(unittest.TestCase):
    def test_split_recordings(self):

        # 11 second audio
        blackbird_audio = np.random.uniform(low=-1.0, high=1.0, size=(11 * 22050,))

        # 6 seconds audio
        sparrow_audio = np.random.uniform(low=-1.0, high=1.0, size=(6 * 22050,))

        pre_df = pd.DataFrame(
            {
                "label": ["blackbird", "sparrow"],
                "audio_buffer": [blackbird_audio, sparrow_audio],
            }
        )

        result_df = split_filter.call(pre_df)

        print(result_df)

        # there are 3 results
        self.assertEqual(len(result_df.index), 3)

        # each buffer is 5s long
        self.assertEqual(len(result_df.iloc[0]["audio_buffer"]), 5 * 22050)

        # two are blackbirds
        self.assertEqual(len(result_df.loc[result_df["label"] == "blackbird"]), 2)

        # one is a sparrow
        self.assertEqual(len(result_df.loc[result_df["label"] == "sparrow"]), 1)


"""
expected_data = librosa.load("fixtures/split_filter_expected.wav")
expected_data = noise_reduction.__split_audio(expected_data, 5)
actual_data = librosa.load("fixtures/split_filter_test.wav")
actual_data = split_filter.__split_buffer(actual_data[0], 22050, 5)
actual_data = split_filter.__filter_chunks(actual_data)
print(np.array_equal(expected_data, actual_data)) # returns true
# Uses instead: self.assertTrue(np.array_equal(expected_data, actual_data))
"""
