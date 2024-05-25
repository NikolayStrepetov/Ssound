import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy import signal


class Sound:
    def __init__(self, rate=0, data=None):
        self.rate = rate
        self.data = data if data is not None else []

    def get_rate(self):
        return self.rate

    def set_rate(self, rate):
        self.rate = rate

    def get_data(self):
        return self.data

    def set_data(self, data):
        self.data = data


def read_audio(file_path):
    """
    Read an audio file.

    Parameters:
    file_path : str
        Path to the .WAV file.

    Returns:
    sound : Sound
        Sound object containing the sampling rate and audio data.
    """

    rate, data = wav.read(file_path)
    return Sound(rate, data)


def write_audio(sound, file_path):
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

    wav.write(file_path, sound.get_rate(), sound.get_data())


def merge_audio(sound1, sound2, first_file_gain=1, second_file_gain=1):
    """
    Merge two audio files with specified gains.

    Parameters:
    sound1 : Sound
        First Sound object.
    sound2 : Sound
        Second Sound object.
    first_file_gain : float
        Gain for the first audio file.
    second_file_gain : float
        Gain for the second audio file.

    Returns:
    sound : Sound
        Sound object containing the sampling rate and audio data.
    """

    data1 = first_file_gain * sound1.get_data()
    data2 = second_file_gain * sound2.get_data()
    rate1 = sound1.get_rate()
    rate2 = sound2.get_rate()

    if rate1 != rate2:
        raise ValueError("Sampling rates of the audio files do not match")

    merged_data = data1 + data2

    return Sound(rate1, merged_data)


def concatenate_audio(sound1, sound2):
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

    data1 = sound1.get_data()
    data2 = sound2.get_data()
    rate1 = sound1.get_rate()
    rate2 = sound2.get_rate()

    if rate1 != rate2:
        raise ValueError("Sampling rates of the audio files do not match")

    concatenated_data = np.concatenate((data1, data2), axis=0)

    return Sound(rate1, concatenated_data)


def split_audio(sound, segment_length, overlap, remainder):
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
    None
    """

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


def change_volume(sound, gain):
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

    amplified_data = sound.get_data() * gain
    amplified_data = np.clip(amplified_data, -32768, 32767).astype(np.int16)

    return Sound(sound.get_rate(), amplified_data)


def plot_spectrogram(sound, channel=1, save_plot=False, output_file=None):
    """
    Plot a spectrogram of an audio file.

    Parameters:
    sound : Sound
        Sound object containing the audio data.
    channel : int
        Channel to plot (1 for first channel, 2 for second channel if stereo).
    save_plot : bool
        Whether to save the plot as an image file.
    output_file : str, optional
        Path to save the image file. If None, saves with the same name as the audio file with .png extension.

    Returns:
    None
    """

    rate = sound.get_rate()
    data = sound.get_data()

    if len(data.shape) == 1:
        frequencies_arr, times_arr, sxx = signal.spectrogram(data, rate)
    else:
        frequencies_arr, times_arr, sxx = signal.spectrogram(data[:, channel-1], rate)

    sxx = np.where(sxx == 0, 1e-12, sxx)

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times_arr, frequencies_arr, 10 * np.log10(sxx))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(f'Spectrogram, channel {channel}')
    plt.colorbar(label='Amplitude [dB]')
    plt.tight_layout()
    if save_plot:
        if output_file is None:
            output_file = "spectrogram.png"
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()


def subtract_audio(sound1, sound2):
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

    data1 = sound1.get_data()
    data2 = sound2.get_data()
    rate1 = sound1.get_rate()
    rate2 = sound2.get_rate()

    if rate1 != rate2:
        raise ValueError("Sampling rates of the audio files do not match")

    max_length = max(len(data1), len(data2))
    pad_width = ((0, max_length - len(data1)), (0, 0))
    data1 = np.pad(data1, pad_width, mode='constant', constant_values=0)
    pad_width = ((0, max_length - len(data2)), (0, 0))
    data2 = np.pad(data2, pad_width, mode='constant', constant_values=0)

    subtracted_data = data1 - data2

    return Sound(rate1, subtracted_data)


"""SNR calculation"""


def calculate_snr(signal_data, noise_data):
    signal_power = np.sum(signal_data ** 2) / len(signal_data)
    noise_power = np.sum(noise_data ** 2) / len(noise_data)
    snr = 10 * np.log10(abs(signal_power / noise_power))
    return snr


"""SDR calculation"""


def calculate_sdr(original_signal, processed_signal):
    distortion = processed_signal - original_signal
    signal_power = np.sum(original_signal ** 2)
    distortion_power = np.sum(distortion ** 2)
    sdr = 10 * np.log10(signal_power / distortion_power)
    return sdr
