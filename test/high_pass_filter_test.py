import unittest
import librosa
import scipy.io.wavfile

class HighPassTest(unittest.TestCase):
    def test_high_pass_filter(self):
        expected_audio = librosa.load("fixtures/high_pass_filtered.wav")
        audio_data = librosa.load("fixtures/chirp.wav")
        actual_audio = high_pass_filter.process(audio_data)
        

'''
testing = HighPassTest()
output = testing.test_high_pass_filter()
scipy.io.wavfile.write("fixtures/high_pass_chirp.wav", output[1], output[0])
'''