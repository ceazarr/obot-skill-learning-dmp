import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d  # <-- Added this import


def find_grip_release_indices_pick_and_place(all_frames, close_thresh, open_thresh, wrist_thresh, hold_duration):

    is_gripping = False
    grip_frame_index = -1
    release_frame_index = -1

    potential_grip_start_time = None
    potential_release_start_time = None

    for i, frame in enumerate(all_frames):
        timestamp = frame.get("timestamp")
        pinch_dist = frame.get("pinch_m", -1)
        wrist_dist = frame.get("wrist_to_cup_m", -1)

        if timestamp is None or pinch_dist < 0 or wrist_dist < 0:
            potential_grip_start_time = None
            potential_release_start_time = None
            continue

        if not is_gripping:
            if pinch_dist <= close_thresh and wrist_dist <= wrist_thresh:
                if potential_grip_start_time is None:
                    potential_grip_start_time = timestamp
                elif timestamp - potential_grip_start_time >= hold_duration:
                    is_gripping = True
                    grip_frame_index = i
                    potential_grip_start_time = None
                    print(f"Grip confirmed at frame {i} (held for {hold_duration}s)")
            else:
                potential_grip_start_time = None
        else:
            if pinch_dist >= open_thresh or wrist_dist >= wrist_thresh:
                if potential_release_start_time is None:
                    potential_release_start_time = timestamp
                elif timestamp - potential_release_start_time >= hold_duration:
                    is_gripping = False
                    release_frame_index = i
                    potential_release_start_time = None
                    print(f"Release confirmed at frame {i} (held for {hold_duration}s)")
            else:
                potential_release_start_time = None

    return grip_frame_index, release_frame_index


def interpolate_nan_values(t, pos):

    T, J, D = pos.shape
    pos_inter = np.copy(pos)

    sort_idx = np.argsort(t)
    t_sorted = t[sort_idx]
    pos_sorted = pos_inter[sort_idx]

    pos_inter_sorted = np.copy(pos_sorted)

    for j in range(J):
        for d in range(D):
            series = pos_sorted[:, j, d]
            valid_indices = np.where(np.isfinite(series))[0]

            if len(valid_indices) < 2:
                # Fallback: Forward fill
                last_valid_val = np.nan
                for i in range(T):
                    if np.isfinite(series[i]):
                        last_valid_val = series[i]
                    elif np.isfinite(last_valid_val):
                        pos_inter_sorted[i, j, d] = last_valid_val

                # Fallback: Backward fill
                first_valid_val = np.nan
                for i in range(T - 1, -1, -1):
                    if np.isfinite(pos_inter_sorted[i, j, d]):
                        first_valid_val = pos_inter_sorted[i, j, d]
                    elif np.isfinite(first_valid_val):
                        pos_inter_sorted[i, j, d] = first_valid_val
                continue

            interp_func = interp1d(
                t_sorted[valid_indices],
                series[valid_indices],
                kind='linear',
                bounds_error=False,
                fill_value=(series[valid_indices[0]], series[valid_indices[-1]])
            )

            all_indices = np.arange(T)
            pos_inter_sorted[all_indices, j, d] = interp_func(t_sorted[all_indices])

    reverse_sort_idx = np.argsort(sort_idx)
    pos_inter = pos_inter_sorted[reverse_sort_idx]

    pos_inter[np.isnan(pos_inter)] = 0.0
    return pos_inter


def plot_grip_segmentation(input_filepath, output_dir):
    print(f"Processing and plotting '{input_filepath}'...")

    try:
        with open(input_filepath, 'r') as f:
            all_frames = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        return

    if not all_frames:
        print("Error: The recording is empty.")
        return

    # --- Segmentation Thresholds ---
    PINCH_CLOSE_THRESH = 0.08
    PINCH_OPEN_THRESH = 0.12
    WRIST_TO_CUP_THRESH = 0.20
    HOLD_DURATION = 0.15

    grip_idx, release_idx = find_grip_release_indices_pick_and_place(
        all_frames,
        PINCH_CLOSE_THRESH,
        PINCH_OPEN_THRESH,
        WRIST_TO_CUP_THRESH,
        HOLD_DURATION
    )

    if grip_idx == -1 or release_idx == -1 or release_idx <= grip_idx:
        print(f"Error: Could not find valid grip/release (grip={grip_idx}, release={release_idx}). Plotting raw data.")
        grip_idx, release_idx = -1, -1
    else:
        print(f"Segmentation found: Grip at frame {grip_idx}, Release at frame {release_idx}")

    timestamps_raw = []
    pinch_distances_raw = []
    wrist_distances_raw = []

    for f in all_frames:
        ts = f.get("timestamp")
        pinch = f.get("pinch_m", -1)
        wrist = f.get("wrist_to_cup_m", -1)

        timestamps_raw.append(ts if ts is not None else np.nan)
        pinch_distances_raw.append(pinch if pinch >= 0 else np.nan)
        wrist_distances_raw.append(wrist if wrist >= 0 else np.nan)

    t = np.array(timestamps_raw, dtype=np.float64)
    pinch_data = np.array(pinch_distances_raw, dtype=np.float64)
    wrist_data = np.array(wrist_distances_raw, dtype=np.float64)

    valid_t_idx = ~np.isnan(t)
    t = t[valid_t_idx]
    pinch_data = pinch_data[valid_t_idx]
    wrist_data = wrist_data[valid_t_idx]

    if len(t) < 2:
        print("Not enough valid data to plot.")
        return


    T = len(t)
    pos_dummy = np.full((T, 2, 1), np.nan, dtype=np.float32)
    pos_dummy[:, 0, 0] = pinch_data
    pos_dummy[:, 1, 0] = wrist_data

    print("Interpolating noisy data...")
    pos_clean = interpolate_nan_values(t, pos_dummy)

    # Extract the cleaned 1D signals
    pinch_distances_clean = pos_clean[:, 0, 0]
    wrist_distances_clean = pos_clean[:, 1, 0]
    print("Interpolation complete.")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)


    ax1.plot(t, pinch_data, 'o', color='blue', alpha=0.2, markersize=2, label="Pinch (Raw)")

    ax1.plot(t, pinch_distances_clean, color='blue', alpha=0.8, label="Pinch (Interpolated)")
    ax1.axhline(PINCH_CLOSE_THRESH, color='green', linestyle='--', label=f"Pinch Close ({PINCH_CLOSE_THRESH:.2f} m)")
    ax1.axhline(PINCH_OPEN_THRESH, color='red', linestyle='--', label=f"Pinch Open ({PINCH_OPEN_THRESH:.2f} m)")

    ax2.plot(t, wrist_data, 'o', color='orange', alpha=0.2, markersize=2, label="Wrist (Raw)")
    ax2.plot(t, wrist_distances_clean, color='orange', alpha=0.8, label="Wrist (Interpolated)")
    ax2.axhline(WRIST_TO_CUP_THRESH, color='purple', linestyle='--', label=f"Wrist Thresh ({WRIST_TO_CUP_THRESH:.2f} m)")

    if grip_idx != -1 and release_idx != -1:
        grip_time = all_frames[grip_idx]['timestamp']
        release_time = all_frames[release_idx]['timestamp']

        for ax in [ax1, ax2]:
            ymin, ymax = ax.get_ylim()

            ax.axvline(grip_time, color='green', linestyle='-.', label=f"GRIP (Start Moving)")
            ax.text(grip_time, ymax*0.9, " Reaching", horizontalalignment='right', color='gray')

            ax.axvline(release_time, color='red', linestyle='-.', label=f"RELEASE (Start Retreating)")
            ax.text(release_time, ymax*0.9, " Moving (Pick/Place)", horizontalalignment='left', color='black')

            ax.axvspan(min(t), grip_time, facecolor='gray', alpha=0.1)
            ax.axvspan(grip_time, release_time, facecolor='green', alpha=0.15)
            ax.axvspan(release_time, max(t), facecolor='gray', alpha=0.1)

    ax1.set_title("Pinch and Wrist Distance Segmentation (with Interpolation)")
    ax1.set_ylabel("Pinch Distance (m)")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Wrist Distance (m)")
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

if __name__ == "__main__":

    input_file = "/home/ceazar/src/skilllearninglib/data/recordings/3DTest_keypoints3d_pickAndPlace1.json"

    output_dir = "./"

    plot_grip_segmentation(input_file, output_dir)

    plt.show()