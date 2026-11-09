from pathlib import Path
import numpy as np
import pandas as pd


# =========================================================
# 1. Configuration
# =========================================================

# Folder containing the original PEMS03.npz, PEMS04.npz, etc.
data_dir = Path(".")

# Folder for the processed datasets and timestamp-index CSV files.
save_root = Path(".")

dataset_names = ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]
window_sizes = [24, 36, 48]

steps_per_day = 288
sampling_minutes = 5


# Published dataset metadata and the start timestamp used to reconstruct
# the benchmark's ordered 5-minute index.
DATASET_TIME_CONFIG = {
    "PEMS03": {
        "num_nodes": 358,
        "num_steps": 26208,
        "start": "2018-09-01 00:00:00",
        "reported_range": "2018-09-01 to 2018-11-30",
        "index_note": "Continuous 5-minute index reconstructed from the published start date.",
    },
    "PEMS04": {
        "num_nodes": 307,
        "num_steps": 16992,
        "start": "2018-01-01 00:00:00",
        "reported_range": "2018-01-01 to 2018-02-28",
        "index_note": "Continuous 5-minute index reconstructed from the published start date.",
    },
    "PEMS07": {
        "num_nodes": 883,
        "num_steps": 28224,
        "start": "2017-05-01 00:00:00",
        "reported_range": "2017-05-01 to 2017-08-31",
        "index_note": (
            "Benchmark-reconstructed continuous index. The published date range "
            "covers 123 days, but 28,224 samples represent only 98 days. "
            "The commonly used LibCity conversion starts at 2017-05-01 and "
            "assigns the first 28,224 consecutive 5-minute timestamps."
        ),
    },
    "PEMS08": {
        "num_nodes": 170,
        "num_steps": 17856,
        "start": "2016-07-01 00:00:00",
        "reported_range": "2016-07-01 to 2016-08-31",
        "index_note": "Continuous 5-minute index reconstructed from the published start date.",
    },
}


# =========================================================
# 2. Build calendar timestamps and time features
# =========================================================

def build_timestamps(dataset_name, num_steps):
    """
    Reconstruct an ordered 5-minute timestamp index.

    time_of_day:
        Fraction of the day in [0, 1).

    day_of_week:
        Monday=0, Tuesday=1, ..., Sunday=6.

    Important:
        The standard NPZ files contain ordered traffic observations but do not
        provide an original timestamp for every row. Therefore, these timestamps
        are reconstructed from the published start date and 5-minute interval.

        PEMS07 needs special attention: its reported range is May-August 2017
        (123 days), whereas 28,224 observations correspond to 98 days.
        This script follows the commonly used LibCity reconstruction convention:
        start at 2017-05-01 00:00 and assign consecutive 5-minute timestamps.
        The resulting index ends at 2017-08-06 23:55. It must not be described as
        a verified row-level timestamp mapping from the original PeMS records.
    """
    config = DATASET_TIME_CONFIG[dataset_name]
    start = pd.Timestamp(config["start"])

    timestamps = pd.date_range(
        start=start,
        periods=num_steps,
        freq=f"{sampling_minutes}min",
    )

    if dataset_name != "PEMS07":
        expected_days = num_steps // steps_per_day
        if num_steps % steps_per_day != 0:
            raise ValueError(
                f"{dataset_name}: {num_steps} is not an integer number of days."
            )

        expected_end = start + pd.Timedelta(days=expected_days) - pd.Timedelta(
            minutes=sampling_minutes
        )
        if timestamps[-1] != expected_end:
            raise RuntimeError(
                f"{dataset_name}: generated end timestamp {timestamps[-1]} "
                f"does not match expected {expected_end}."
            )

    return timestamps


def resolve_data_file(dataset_name):
    """Find either ./PEMS03.npz or ./PEMS03/PEMS03.npz."""
    candidates = [
        data_dir / f"{dataset_name}.npz",
        data_dir / dataset_name / f"{dataset_name}.npz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = "\\n".join(str(p.resolve()) for p in candidates)
    raise FileNotFoundError(
        f"Cannot find {dataset_name}.npz. Checked:\\n{searched}"
    )


def load_data(file_path, dataset_name):
    with np.load(file_path, allow_pickle=False) as raw_data:
        # First feature: traffic flow.
        flow = raw_data["data"][..., :1].astype(np.float32)

    num_steps, num_nodes, _ = flow.shape
    config = DATASET_TIME_CONFIG[dataset_name]

    if num_steps != config["num_steps"]:
        raise ValueError(
            f"{dataset_name}: expected {config['num_steps']} time steps, "
            f"but found {num_steps}."
        )
    if num_nodes != config["num_nodes"]:
        raise ValueError(
            f"{dataset_name}: expected {config['num_nodes']} nodes, "
            f"but found {num_nodes}."
        )

    timestamps = build_timestamps(dataset_name, num_steps)

    # Calendar-based time of day: [0, 1).
    minutes_of_day = timestamps.hour * 60 + timestamps.minute
    time_of_day_1d = (
        minutes_of_day.to_numpy(dtype=np.float32) / (24 * 60)
    )

    # Actual weekday derived from the calendar date:
    # Monday=0, ..., Sunday=6.
    day_of_week_1d = timestamps.dayofweek.to_numpy(dtype=np.float32)

    time_of_day = np.broadcast_to(
        time_of_day_1d[:, None, None],
        (num_steps, num_nodes, 1),
    )
    day_of_week = np.broadcast_to(
        day_of_week_1d[:, None, None],
        (num_steps, num_nodes, 1),
    )

    # Save the timestamp mapping for auditing and later alignment.
    timestamp_df = pd.DataFrame({
        "time_idx": np.arange(num_steps, dtype=np.int32),
        "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "time_of_day": time_of_day_1d,
        "day_of_week": day_of_week_1d.astype(np.int8),
    })

    timestamp_path = save_root / f"{dataset_name}_time_index.csv"
    timestamp_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp_df.to_csv(
        timestamp_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nDataset: {dataset_name}")
    print(f"  Data shape: {flow.shape}")
    print(f"  Reported date range: {config['reported_range']}")
    print(f"  Reconstructed index: {timestamps[0]} -> {timestamps[-1]}")
    print(f"  First day of week: {timestamps[0].day_name()}")
    print(f"  Time index saved to: {timestamp_path.resolve()}")
    print(f"  Note: {config['index_note']}")

    # [T, N, 3]: traffic flow, time of day, day of week.
    return np.concatenate(
        [flow, time_of_day, day_of_week],
        axis=-1,
    )


# =========================================================
# 3. Generate and save one split using its window start indices
# =========================================================

def save_split(
    data,
    starts,
    input_window,
    output_window,
    save_path,
):
    num_samples = len(starts)
    num_nodes = data.shape[1]

    x = np.empty(
        (num_samples, input_window, num_nodes, 3),
        dtype=np.float32,
    )
    y = np.empty(
        (num_samples, output_window, num_nodes, 1),
        dtype=np.float32,
    )

    for index, start in enumerate(starts):
        input_end = start + input_window
        output_end = input_end + output_window

        x[index] = data[start:input_end]
        y[index] = data[input_end:output_end, :, :1]

    np.savez_compressed(save_path, x=x, y=y)

    print(f"  {save_path.name}: x={x.shape}, y={y.shape}")


# =========================================================
# 4. Process one window length
# =========================================================

def process_window(data, dataset_name, window):
    input_window = window
    output_window = window

    # To use a fixed 12-step prediction horizon, replace the line above with:
    # output_window = 12

    num_samples = (
        data.shape[0] - input_window - output_window + 1
    )

    if num_samples < 1:
        raise ValueError(
            f"{dataset_name}: not enough observations for "
            f"input={input_window}, output={output_window}."
        )

    # Chronological split of sliding-window start indices.
    train_size = int(num_samples * 0.6)
    remaining_size = num_samples - train_size
    test_size = remaining_size // 2
    val_size = remaining_size - test_size

    train_end = train_size
    val_end = train_size + val_size

    train_starts = np.arange(0, train_end)
    val_starts = np.arange(train_end, val_end)
    test_starts = np.arange(val_end, num_samples)

    output_dir = save_root / f"{dataset_name}_{window}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to: {output_dir.resolve()}")
    print(
        f"  Total samples={num_samples}, "
        f"train={train_size}, val={val_size}, test={test_size}"
    )

    for split_name, starts in (
        ("train", train_starts),
        ("val", val_starts),
        ("test", test_starts),
    ):
        save_split(
            data=data,
            starts=starts,
            input_window=input_window,
            output_window=output_window,
            save_path=output_dir / f"{split_name}.npz",
        )


# =========================================================
# 5. Process all datasets
# =========================================================

def main():
    save_root.mkdir(parents=True, exist_ok=True)

    # Check all source files before starting.
    source_files = {
        dataset_name: resolve_data_file(dataset_name)
        for dataset_name in dataset_names
    }

    for dataset_name in dataset_names:
        file_path = source_files[dataset_name]
        data = load_data(file_path, dataset_name)

        for window in window_sizes:
            process_window(data, dataset_name, window)

    print("\nAll datasets have been processed successfully.")


if __name__ == "__main__":
    main()
