import numpy as np
from .sound import Sound
from .utils import audio_length_alignment
from scipy.fftpack import dct
from scipy.signal.windows import hamming


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


def calculate_mfcc(sound, n_mfcc=13, n_fft=512, hop_length=256, n_mels=23, fmin=0, fmax=None):
    """
    Calculate the MFCC (Mel-frequency cepstral coefficients) of an audio file.

    Parameters:
    sound : Sound
        Sound object containing the audio data.
    n_mfcc : int
        Number of MFCCs to return (default is 13).
    n_fft : int
        Number of FFT points (default is 512).
    hop_length : int
        Number of samples between successive frames (default is 256).
    n_mels : int
        Number of Mel bands to use (default is 23).
    fmin : int
        Minimum frequency (default is 0).
    fmax : int
        Maximum frequency (default is None, which will be half of the sampling rate).

    Returns:
    mfccs : np.array
        Array of MFCCs for the audio data.
    """
    data = sound.get_data()
    rate = sound.get_rate()

    if fmax is None:
        fmax = rate // 2  # Nyquist frequency

    # Step 1: Compute the short-time Fourier transform (STFT)
    frames = np.array([data[i:i + n_fft] * hamming(n_fft) for i in range(0, len(data) - n_fft, hop_length)])

    # Compute the magnitude spectrum
    spectrum = np.abs(np.fft.rfft(frames, n=n_fft))

    # Step 2: Apply the Mel filter bank
    mel_points = np.linspace(fmin, fmax, n_mels + 2)
    mel_filter_bank = np.zeros((n_mels, spectrum.shape[1]))

    # Mel scale conversion
    for i in range(1, n_mels + 1):
        left = mel_points[i - 1]
        center = mel_points[i]
        right = mel_points[i + 1]

        mel_filter_bank[i - 1] = np.maximum(0, (np.linspace(0, 1, int(center - left)) * (center - left))[
                                   np.newaxis] + np.maximum(0, (np.linspace(1, 0, int(right - center)) * (right - center))))

    # Apply filter bank to the magnitude spectrum
    mel_spectrum = np.dot(mel_filter_bank, spectrum)

    # Step 3: Apply logarithm
    log_mel_spectrum = np.log(mel_spectrum + 1e-6)

    # Step 4: Compute the Discrete Cosine Transform (DCT)
    mfccs = dct(log_mel_spectrum, type=2, axis=-1, norm='ortho')[:, :n_mfcc]

    return mfccs