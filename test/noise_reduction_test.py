import unittest
import soundfile as sf
from francis import noise_reduction
import librosa


class NoiseReductionTest(unittest.TestCase):
    def test_audio_splitting(self):
        # this is a lame test but I couldn't think of anything else
        # it works but it produces slightly different results each time
        # but the same overall effect

        expected_audio = librosa.load("test/fixtures/reduced_chirp.wav")

        audio_data = librosa.load("test/fixtures/chirp.wav")
        actual_audio = noise_reduction.process(audio_data, 0.5)
