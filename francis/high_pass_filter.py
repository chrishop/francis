import librosa
import scipy.signal as sg


def process(audio_data):
    noisy_data, sf = audio_data
    b, a = sg.butter(4, 1000.0 / (sf / 2.0), "high")
    filtered_data = sg.filtfilt(b, a, noisy_data)
    return (
        filtered_data,
        sf,
    )  # return filtered_data in same tuple format of audio_data


"""
clean_data, sf = librosa.load("fixtures/noisy.mp3")
scipy.io.wavfile.write("fixtures/high_pass_noisy.wav", sf, clean_data)
b, a = sg.butter(4, 1000. / (sf / 2.), 'high')
filtered_data = sg.filtfilt(b, a, clean_data)
scipy.io.wavfile.write("fixtures/high_pass_filtered.wav", sf, filtered_data)
"""
