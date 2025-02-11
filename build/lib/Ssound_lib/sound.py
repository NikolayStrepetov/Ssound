import numpy as np


class Sound:
    def __init__(self, rate: int = 0, data: np.ndarray = None):
        if not isinstance(rate, int) or rate <= 0:
            raise ValueError("Rate must be a positive integer")

        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array or None")

        self.rate = rate
        self.data = data if data is not None else np.array([])

    def get_rate(self) -> int:
        return self.rate

    def set_rate(self, rate: int) -> None:
        if not isinstance(rate, int) or rate <= 0:
            raise ValueError("Rate must be a positive integer")

        self.rate = rate

    def get_data(self) -> np.ndarray:
        return self.data

    def set_data(self, data: np.ndarray) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array")

        self.data = data
