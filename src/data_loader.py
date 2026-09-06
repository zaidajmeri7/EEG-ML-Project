from pathlib import Path
import pickle
import numpy as np


def load_deap(
    data_dir: str | Path,
    subject_ids: list[int] | None = None
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Load DEAP dataset.

    Returns
    -------
    data : np.ndarray
        Shape: (n_trials, n_channels, n_samples)

    labels : np.ndarray
        Shape: (n_trials, n_labels)

    metadata : list[dict]
        Information about dataset, subject, and trial.
    """

    data_dir = Path(data_dir)

    if subject_ids is None:
        subject_ids = list(range(1, 41))

    all_data = []
    all_labels = []
    metadata = []

    for subject_id in subject_ids:

        file_path = data_dir / f"s{subject_id:02d}.dat"

        if not file_path.exists():
            print(f"Warning: {file_path} not found. Skipping.")
            continue

        with open(file_path, "rb") as f:
            subject = pickle.load(f, encoding="latin1")

        data = np.asarray(subject["data"], dtype=np.float64)
        labels = np.asarray(subject["labels"], dtype=np.float64)

        # DEAP:
        # First 32 channels = EEG
        eeg_data = data[:, :32, :]

        all_data.append(eeg_data)
        all_labels.append(labels)

        for trial_idx in range(len(eeg_data)):
            metadata.append({
                "dataset": "DEAP",
                "subject": subject_id,
                "trial": trial_idx + 1
            })

        print(
            f"Loaded subject {subject_id}: "
            f"{eeg_data.shape}"
        )

    if not all_data:
        raise FileNotFoundError(
            "No DEAP files were found."
        )

    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return data, labels, metadata