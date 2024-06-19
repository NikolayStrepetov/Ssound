import numpy as np
from .sound import Sound
from .utils import audio_length_alignment


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
