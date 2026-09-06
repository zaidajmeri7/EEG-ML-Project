import numpy as np
from scipy.signal import welch


FREQUENCY_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def extract_bandpower_features(
    data: np.ndarray,
    fs: int
) -> tuple[np.ndarray, list[str]]:
    """
    Convert EEG trials into dataset-independent features.

    Input
    -----
    data:
        Shape (trials, channels, samples)

    Output
    ------
    features:
        Shape (trials, 10)

    feature_names:
        Names of the 10 features.
    """

    feature_rows = []

    feature_names = []

    for band_name in FREQUENCY_BANDS:
        feature_names.append(f"{band_name}_mean")
        feature_names.append(f"{band_name}_std")

    for trial in data:

        channel_bandpowers = []

        for channel in trial:

            frequencies, power = welch(
                channel,
                fs=fs,
                nperseg=min(256, len(channel))
            )

            bandpowers = []

            for low, high in FREQUENCY_BANDS.values():

                mask = (
                    (frequencies >= low) &
                    (frequencies < high)
                )

                if not np.any(mask):
                    band_power = 0.0
                else:
                    band_power = np.mean(power[mask])

                bandpowers.append(band_power)

            channel_bandpowers.append(bandpowers)

        channel_bandpowers = np.asarray(channel_bandpowers)

        features = []

        for band_index in range(len(FREQUENCY_BANDS)):

            values = channel_bandpowers[:, band_index]

            features.append(np.mean(values))
            features.append(np.std(values))

        feature_rows.append(features)

    return np.asarray(feature_rows), feature_names