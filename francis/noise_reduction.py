import noisereduce as nr


def process(audio_data):
    data, rate = audio_data
    backround_noise, rate = __find_backround_noise(data)
    return nr.reduce_noise(
        audio_clip=data,
        noise_clip=backround_noise,
        verbose=True)


def __find_backround_noise(audio_data):
    """finds a 5s of quietest part of the audio"""
    
    return audio_data
