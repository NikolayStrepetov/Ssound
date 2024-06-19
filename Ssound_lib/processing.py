import numpy as np
from .sound import Sound
from .utils import audio_length_alignment


def change_volume(sound: Sound, gain: float) -> Sound:
    """
    Increase the volume of an audio file.

    Parameters:
    sound : Sound
        Sound object containing the audio data.
    gain : float
        Gain factor to apply.

    Returns:
    sound : Sound
        Sound object containing the amplified audio data.
    """
    if not isinstance(sound, Sound):
        raise TypeError("Expected 'sound' to be an instance of Sound")
    if not isinstance(gain, (float, int)):
        raise TypeError("Expected 'gain' to be a float or int")
    if gain < 0:
        raise ValueError("Expected 'gain' must be positive or 0")

    amplified_data = sound.get_data() * gain
    amplified_data = np.clip(amplified_data, -32768, 32767).astype(np.int16)
    return Sound(sound.get_rate(), amplified_data)


def merge_audio(sound1: Sound, sound2: Sound) -> Sound:
    """
    Merge two audio files with specified gains.

    Parameters:
    sound1 : Sound
        First Sound object.
    sound2 : Sound
        Second Sound object.

    Returns:
    sound : Sound
        Sound object containing the sampling rate and audio data.
    """
    if not isinstance(sound1, Sound) or not isinstance(sound2, Sound):
        raise TypeError("Both inputs must be instances of Sound")

    data1 = sound1.get_data()
    data2 = sound2.get_data()
    rate1 = sound1.get_rate()
    rate2 = sound2.get_rate()

    if rate1 != rate2:
        raise ValueError("Sampling rates of the audio files do not match")

    data1, data2 = audio_length_alignment(data1, data2)
    merged_data = data1 + data2
    return Sound(rate1, merged_data)


def concatenate_audio(sound1: Sound, sound2: Sound) -> Sound:
    """
    Concatenate two audio files.

    Parameters:
    sound1 : Sound
        First Sound object.
    sound2 : Sound
        Second Sound object.

    Returns:
    sound : Sound
        Sound object containing the sampling rate and audio data.
    """
    if not isinstance(sound1, Sound) or not isinstance(sound2, Sound):
        raise TypeError("Both inputs must be instances of Sound")

    data1 = sound1.get_data()
    data2 = sound2.get_data()
    rate1 = sound1.get_rate()
    rate2 = sound2.get_rate()

    if rate1 != rate2:
        raise ValueError("Sampling rates of the audio files do not match")

    concatenated_data = np.concatenate((data1, data2), axis=0)
    return Sound(rate1, concatenated_data)


def split_audio(sound: Sound, segment_length: float, overlap: float, remainder: bool) -> list:
    """
    Split an audio file into segments of a given length with overlap.

    Parameters:
    sound : Sound
        Sound object containing the audio data to split.
    segment_length : float
        Length of each segment in seconds.
    overlap : float
        Overlap between segments in seconds.
    remainder : bool
        Whether to keep the remainder segment if it is shorter than segment_length.

    Returns:
    segments : list
        List of Sound objects, each representing a segment of the original audio.
    """
    if not isinstance(sound, Sound):
        raise TypeError("Expected 'sound' to be an instance of Sound")
    if not isinstance(segment_length, (float, int)) or not isinstance(overlap, (float, int)):
        raise TypeError("Expected 'segment_length' and 'overlap' to be floats or ints")
    if segment_length <= 0 or overlap < 0:
        raise ValueError("segment_length must be positive and overlap cannot be negative")

    rate = sound.get_rate()
    data = sound.get_data()

    segment_length = int(rate * segment_length)
    overlap_length = int(rate * overlap)

    segments = []
    start = 0
    while start < len(data):
        end = min(start + segment_length, len(data))
        segment = data[start:end]
        if remainder or len(segment) == segment_length:
            segments.append(Sound(rate, segment))
        start += segment_length - overlap_length

    return segments


def subtract_audio(sound1: Sound, sound2: Sound) -> Sound:
    """
    Subtract a second audio file from the first.

    Parameters:
    sound1 : Sound
        First Sound object.
    sound2 : Sound
        Second Sound object.

    Returns:
    sound : Sound
        Sound object containing the sampling rate and audio data.
    """
    if not isinstance(sound1, Sound) or not isinstance(sound2, Sound):
        raise TypeError("Both inputs must be instances of Sound")

    data1 = sound1.get_data()
    data2 = sound2.get_data()
    rate1 = sound1.get_rate()
    rate2 = sound2.get_rate()

    if rate1 != rate2:
        raise ValueError("Sampling rates of the audio files do not match")

    data1, data2 = audio_length_alignment(data1, data2)
    subtracted_data = data1 - data2
    return Sound(rate1, subtracted_data)
