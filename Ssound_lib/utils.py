import numpy as np


def audio_length_alignment(data1: np.ndarray, data2: np.ndarray) -> (np.ndarray, np.ndarray):
    """
        Align the lengths of two audio signals by padding the shorter one with zeros.

        Parameters:
        data1 : np.array
            Array containing the first audio data.
        data2 : np.array
            Array containing the second audio data.

        Returns:
        aligned_data1 : np.array
            First audio data array padded to match the length of the longer array.
        aligned_data2 : np.array
            Second audio data array padded to match the length of the longer array.
    """

    if not isinstance(data1, np.ndarray) or not isinstance(data2, np.ndarray):
        raise TypeError("Both data1 and data2 must be numpy arrays")

    if data1.ndim != data2.ndim:
        raise ValueError("Both input arrays must have the same number of dimensions")

    max_length = max(len(data1), len(data2))

    pad_width = ((0, max_length - len(data1)), (0, 0))
    aligned_data1 = np.pad(data1, pad_width, mode='constant', constant_values=0)

    pad_width = ((0, max_length - len(data2)), (0, 0))
    aligned_data2 = np.pad(data2, pad_width, mode='constant', constant_values=0)

    return aligned_data1, aligned_data2


def hz_to_mel(f):
    """
        Converts the frequency of a sound (Hz) into pitch (Mel).

        Parameters:
        f : float
            Frequency of a sound (Hz).

        Returns:
        result : float
            Sound pitch (Mel).
    """
    return 2595 * np.log10(1 + f / 700)


def mel_to_hz(m):
    """
        Converts pitch of a sound (Mel) into frequency (Hz).

        Parameters:
        m : float
            Sound pitch (Mel).

        Returns:
        result : float
            Frequency of a sound (Hz).
    """
    return 700 * (10**(m / 2595) - 1)
