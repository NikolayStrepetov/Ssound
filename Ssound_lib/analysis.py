import numpy as np
from .sound import Sound
from .utils import audio_length_alignment, hz_to_mel, mel_to_hz
from scipy.signal import stft
import scipy.fftpack as fft


def calculate_snr(clean_sound: Sound, noisy_sound: Sound) -> float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) of two Sound objects.

    Parameters:
    clean_sound : Sound
        Sound object containing the clean audio data.
    noisy_sound : Sound
        Sound object containing the noisy audio data.

    Returns:
    snr : float
        Signal-to-Noise Ratio (SNR) in linear scale (fractions).
    """
    if not isinstance(clean_sound, Sound) or not isinstance(noisy_sound, Sound):
        raise TypeError("Both inputs must be instances of Sound")

    clean_data = clean_sound.get_data()
    noisy_data = noisy_sound.get_data()

    if clean_sound.get_rate() != noisy_sound.get_rate():
        raise ValueError("Sampling rates of the audio files do not match")

    clean_data, noisy_data = audio_length_alignment(clean_data, noisy_data)

    signal_power = np.mean(clean_data ** 2)
    noise_power = np.mean((noisy_data - clean_data) ** 2)

    if noise_power == 0:
        raise ValueError("Noise power is zero, cannot calculate SNR")

    snr = signal_power / noise_power
    return snr


def calculate_sdr(clean_sound: Sound, estimated_sound: Sound) -> float:
    """
    Calculate the Signal-to-Distortion Ratio (SDR) of two Sound objects.

    Parameters:
    clean_sound : Sound
        Sound object containing the clean audio data.
    estimated_sound : Sound
        Sound object containing the estimated audio data.

    Returns:
    sdr : float
        Signal-to-Distortion Ratio (SDR) in linear scale (fractions).
    """
    if not isinstance(clean_sound, Sound) or not isinstance(estimated_sound, Sound):
        raise TypeError("Both inputs must be instances of Sound")

    clean_data = clean_sound.get_data()
    estimated_data = estimated_sound.get_data()

    if clean_sound.get_rate() != estimated_sound.get_rate():
        raise ValueError("Sampling rates of the audio files do not match")

    clean_data, estimated_data = audio_length_alignment(clean_data, estimated_data)

    signal_power = np.mean(clean_data ** 2)
    distortion_power = np.mean((estimated_data - clean_data) ** 2)

    if distortion_power == 0:
        raise ValueError("Distortion power is zero, cannot calculate SDR")

    sdr = signal_power / distortion_power
    return sdr


def calculate_si_sdr(clean_sound: Sound, estimated_sound: Sound) -> float:
    """
    Calculate the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) of two Sound objects.

    Parameters:
    clean_sound : Sound
        Sound object containing the clean audio data.
    estimated_sound : Sound
        Sound object containing the estimated audio data.

    Returns:
    si_sdr : float
        Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in decibels.
    """
    if not isinstance(clean_sound, Sound) or not isinstance(estimated_sound, Sound):
        raise TypeError("Both inputs must be instances of Sound")

    clean_data = clean_sound.get_data()
    estimated_data = estimated_sound.get_data()

    if clean_sound.get_rate() != estimated_sound.get_rate():
        raise ValueError("Sampling rates of the audio files do not match")

    clean_data, estimated_data = audio_length_alignment(clean_data, estimated_data)

    if len(clean_data.shape) > 1:
        si_sdr_list = []
        for channel in range(clean_data.shape[1]):
            si_sdr_list.append(_calculate_channel_si_sdr(clean_data[:, channel], estimated_data[:, channel]))
        return np.mean(si_sdr_list)
    else:
        return _calculate_channel_si_sdr(clean_data, estimated_data)


def _calculate_channel_si_sdr(clean_data: np.array, estimated_data: np.array) -> float:
    """
    Calculate the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) for a single channel.

    Parameters:
    clean_data : np.array
        Array containing the clean audio data for one channel.
    estimated_data : np.array
        Array containing the estimated audio data for one channel.

    Returns:
    si_sdr : float
        Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in decibels for the channel.
    """
    if not isinstance(clean_data, np.ndarray) or not isinstance(estimated_data, np.ndarray):
        raise TypeError("Both inputs must be numpy arrays")
    if len(clean_data) != len(estimated_data):
        raise ValueError("Length of clean_data and estimated_data must be the same")

    alpha = np.dot(estimated_data, clean_data) / np.dot(clean_data, clean_data)
    projected_clean_data = alpha * clean_data

    error = estimated_data - projected_clean_data

    signal_power = np.mean(projected_clean_data ** 2)
    error_power = np.mean(error ** 2)

    if error_power == 0:
        raise ValueError("Error power is zero, cannot calculate SI-SDR")

    si_sdr = 10 * np.log10(signal_power / error_power)

    return si_sdr


def get_mel_filters(sample_rate, n_fft, n_mels=40):
    """
        Creating of Mel-filters.

        Parameters:
        sample_rate : int
            Sample rate of an audio file.
        n_fft : float
            Length of the FFT window.
        n_mels : float
            Number of Mel bands.

        Returns:
        filters : np.array
            Mel-filters.
    """
    low_freq_mel = hz_to_mel(0)
    high_freq_mel = hz_to_mel(sample_rate / 2)

    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bin = np.floor((n_fft + 1) * hz_points / sample_rate)

    filters = np.zeros((n_mels, int(np.floor(n_fft / 2 + 1))))

    for i in range(1, n_mels + 1):
        left = int(bin[i - 1])
        center = int(bin[i])
        right = int(bin[i + 1])

        for j in range(left, center):
            filters[i - 1, j] = (j - bin[i - 1]) / (bin[i] - bin[i - 1])
        for j in range(center, right):
            filters[i - 1, j] = (bin[i + 1] - j) / (bin[i + 1] - bin[i])

    return filters


def calculate_mfcc(sound, n_mfcc=13, n_fft=2048, hop_length=512, n_mels=40):
    """
        Calculation of Mel-frequency cepstral coefficients (MFCC).

        Parameters:
        sound : Sound
            Sound object containing the audio data.
        n_mfcc : float
            Number of MFCCs that will be returned.
        n_fft : float
            Length of the FFT window.
        hop_length : float
            Number of samples between frames.
        n_mels : float
            Number of Mel bands.

        Returns:
        mfcc : np.array
            Sound pitch (Mel).
    """

    signal = sound.get_data()
    sample_rate = sound.get_rate()

    if len(signal.shape) > 1:
        signal = signal.mean(axis=1)

    _, _, Zxx = stft(signal, fs=sample_rate, window='hann', nperseg=n_fft, noverlap=n_fft - hop_length)
    spectrogram = np.abs(Zxx) ** 2

    mel_filters = get_mel_filters(sample_rate, n_fft, n_mels)

    mel_spectrogram = np.dot(mel_filters, spectrogram)

    log_mel_spectrogram = np.log(mel_spectrogram + 1e-6)

    mfcc = fft.dct(log_mel_spectrogram, axis=0, type=2, norm='ortho')[:n_mfcc]

    return mfcc
