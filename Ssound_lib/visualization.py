import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from .sound import Sound


def plot_waveform(sound: Sound, save_plot: bool = False, output_file: str = None) -> None:
    """
    Plot the waveform of an audio file.

    Parameters:
    sound : Sound
        Sound object containing the audio data to plot.

    Returns:
    None
    """

    if not isinstance(sound, Sound):
        raise TypeError("Expected 'sound' to be an instance of Sound")
    if not isinstance(save_plot, bool):
        raise TypeError("'save_plot' must be a bool")
    if not isinstance(output_file, str) and output_file is not None:
        raise TypeError("'output_file' must be a string or None")

    data = sound.get_data()
    rate = sound.get_rate()
    time = np.linspace(0, len(data) / rate, num=len(data))

    plt.figure(figsize=(10, 4))
    plt.plot(time, data)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    if save_plot:
        if output_file is None:
            output_file = "waveform.png"
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()


def plot_spectrogram(sound: Sound, channel: int = 1, save_plot: bool = False, output_file: str = None) -> None:
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

    if not isinstance(sound, Sound):
        raise TypeError("Expected 'sound' to be an instance of Sound")
    if not isinstance(channel, int):
        raise TypeError("'channel' must be an integer")
    if channel < 1:
        raise ValueError("'channel' must be positive")
    if not isinstance(save_plot, bool):
        raise TypeError("'save_plot' must be a bool")
    if not isinstance(output_file, str) and output_file is not None:
        raise TypeError("'output_file' must be a string or None")

    rate = sound.get_rate()
    data = sound.get_data()

    if len(data.shape) == 1:
        frequencies_arr, times_arr, sxx = signal.spectrogram(data, rate)
    else:
        frequencies_arr, times_arr, sxx = signal.spectrogram(data[:, channel - 1], rate)

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
