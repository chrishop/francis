import unittest
import numpy as np
import librosa
import soundfile as sf
import noisereduce


def split_audio(audio_data, seconds):
    amplitudes, sample_rate = audio_data
    chunk_size = sample_rate * seconds
    chunk_number = len(amplitudes) // chunk_size
    block = chunk_number * chunk_size 
    new_amplitudes = amplitudes[:block]
    leftover_amplitudes = amplitudes[block:]
    
    split_audio = np.split(new_amplitudes, chunk_number)
    split_audio[-1] = np.append(split_audio[-1], leftover_amplitudes)
    return split_audio


def write_to_file(splitted_audio, sample_rate):
    for i,elem in enumerate(splitted_audio):
        sf.write(
            f"test/fixtures/split_birdsong_{i}.wav",
            elem,
            sample_rate)
        
        
def select_quietest(split_audio):
    quietest_volume = None
    quietest_index = 0
    for i in range(len(split_audio)):
        average_volume = np.mean(np.abs(split_audio[i]))
        if quietest_volume == None or quietest_volume > average_volume:
            quitest_volume = average_volume
            quietest_index = i
    return split_audio[quietest_index]

def reduce_noise(audio_data):
    splitted_audio = split_audio(audio_data, 2)
    quietest = select_quietest(splitted_audio)
    return noisereduce.reduce_noise(
        audio_clip=audio_data[0], 
        noise_clip=quietest)
    
    

class NoiseReductionTest(unittest.TestCase):
    
    def test_audio_splitting(self):
        audio_data = librosa.load("test/fixtures/sparse_birdsong.mp3")
        reduced = reduce_noise(audio_data)
        sf.write("test/fixtures/reduced.wav", reduced, audio_data[1])
        
        # sf.write("test/fixtures/quietest.wav", quietest, audio_data[1])
        # write_to_file(splitted_audio, audio_data[1])