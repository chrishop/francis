import librosa
import scipy.signal as sg


def process(audio_data):
    noisy_data, sf = audio_data
    b, a = sg.butter(4, 1500.0 / (sf / 2.0), "high")
    filtered_data = sg.filtfilt(b, a, noisy_data)
    return (
        filtered_data,
        sf,
    )  # return filtered_data in same tuple format of audio_data
