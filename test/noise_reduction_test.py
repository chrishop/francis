import unittest
import soundfile as sf
from francis import noise_reduction

    

class NoiseReductionTest(unittest.TestCase):
    
    
    
    def test_audio_splitting(self):
        audio_data = librosa.load("test/fixtures/sparse_birdsong.mp3")
        reduced = noise_reduction.process(audio_data, 2)
        sf.write("test/fixtures/reduced_real.wav", reduced, audio_data[1])
        
       