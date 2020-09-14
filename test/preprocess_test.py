import unittest
import pandas as pd
import numpy as np
from francis import split_filter
from francis import preprocess
import librosa


def noisy_audio(seconds):
    return np.random.uniform(low=-1.0, high=1.0, size=(seconds * 22050,))


def quiet_audio(seconds):
    return np.random.uniform(low=-0.01, high=0.01, size=(seconds * 22050,))


class SplitFilterTest(unittest.TestCase):
    def test_split_recordings(self):

        # 10 second audio
        mixed_audio = np.concatenate([noisy_audio(5), quiet_audio(5)])

        # 11 second audio
        blackbird_audio = noisy_audio(11)

        # 6 seconds audio
        sparrow_audio = noisy_audio(6)

        pre_df = pd.DataFrame(
            {
                "label": ["blackbird", "sparrow", "mixed_noise"],
                "audio_buffer": [blackbird_audio, sparrow_audio, mixed_audio],
            }
        )
        
        pre_df = preprocess.process(pre_df, True)
        result_df = split_filter.call(pre_df)
        
        print(result_df)

        # there are 4 results
        self.assertEqual(len(result_df.index), 4)

        # each buffer is 5s long
        self.assertEqual(len(result_df["audio_buffer"].iloc[0]), 5 * 22050)

        # two are blackbirds
        self.assertEqual(len(result_df.loc[result_df["label"] == "blackbird"]), 2)

        # one is a sparrow
        self.assertEqual(len(result_df.loc[result_df["label"] == "sparrow"]), 1)

        # one is mixed
        self.assertEqual(len(result_df.loc[result_df["label"] == "mixed_noise"]), 1)
