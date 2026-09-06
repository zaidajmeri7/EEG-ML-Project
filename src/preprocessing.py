import numpy as np


def remove_baseline(
    data: np.ndarray,
    fs: int,
    baseline_seconds: float
) -> np.ndarray:
    """
    Remove baseline portion from EEG trials.

    Parameters
    ----------
    data :
        Shape (trials, channels, samples)

    fs :
        Sampling frequency in Hz

    baseline_seconds :
        Length of baseline at beginning of each trial.
    """

    baseline_samples = int(fs * baseline_seconds)

    if baseline_samples >= data.shape[-1]:
        raise ValueError(
            "Baseline duration is longer than the trial."
        )

    return data[:, :, baseline_samples:]


def zscore_channels(data: np.ndarray) -> np.ndarray:
    """
    Z-score each channel independently for each trial.
    """

    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)

    std = np.where(std == 0, 1, std)

    return (data - mean) / std