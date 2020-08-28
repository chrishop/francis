import unittest
import librosa
import numpy as np

import numpy.testing
from francis import high_pass_filter


class HighPassTest(unittest.TestCase):
    def test_high_pass_filter(self):
        expected_audio, _ = librosa.load("test/fixtures/high_pass_filtered.wav")
        audio_data = librosa.load("test/fixtures/high_pass_noisy.wav")
        actual_audio, _ = high_pass_filter.process(audio_data)

        numpy.testing.assert_array_almost_equal(expected_audio, actual_audio)
