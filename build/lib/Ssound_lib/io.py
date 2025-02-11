import os
import scipy.io.wavfile as wav
from .sound import Sound


def read_audio(file_path: str) -> Sound:
    """
    Read an audio file.

    Parameters:
    file_path : str
        Path to the .WAV file.

    Returns:
    sound : Sound
        Sound object containing the sampling rate and audio data.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist")

    try:
        rate, data = wav.read(file_path)
    except ValueError as e:
        raise ValueError(f"Error reading the WAV file: {e}")

    return Sound(rate, data)


def write_audio(sound: Sound, file_path: str) -> None:
    """
    Write an audio file.

    Parameters:
    sound : Sound
        Sound object containing the audio data to write.
    file_path : str
        Path to the output .WAV file.

    Returns:
    None
    """

    try:
        wav.write(file_path, sound.get_rate(), sound.get_data())
    except Exception as e:
        raise ValueError(f"Error writing the WAV file: {e}")
