# input output to files is in its own file 
# so it makes the rest of the code easier to test without side effects

import librosa

"""returns a tuple audio, sample_rate"""
def open_audio(filepath):
    librosa.load(filepath)