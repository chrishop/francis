import numpy as np
import librosa
import noisereduce


def process(audio_data, seconds=2):
    return noisereduce.reduce_noise(
        audio_clip=audio_data[0],
        noise_clip=__backround_noise(audio_data, seconds),
        verbose=False)


def __backround_noise(audio_data, seconds):
    return __quietest(
        __split_audio(
            audio_data,
            seconds
        )
    )


def __quietest(split_amplitudes):
    quietest_volume = None
    quietest_index = 0
    for i in range(len(split_amplitudes)):
        average_volume = np.mean(np.abs(split_amplitudes[i]))
        if quietest_volume == None or quietest_volume > average_volume:
            quitest_volume = average_volume
            quietest_index = i
    return split_amplitudes[quietest_index]
    
    
def __split_audio(audio_data, seconds):
    amplitudes, rate = audio_data
    size_of_elements = rate * seconds
    number_of_elements = len(amplitudes) // size_of_elements
    block = size_of_elements * number_of_elements
    divisible_amplitudes = amplitudes[:block]
    leftover_amplitudes = amplitudes[block:]
    
    return __append_leftover(
       np.split(divisible_amplitudes, number_of_elements),
       leftover_amplitudes 
    )
    
    
def __append_leftover(split_amplitudes, leftover):
    split_amplitudes[-1] = np.append(split_amplitudes[-1], leftover)
    return split_amplitudes
    