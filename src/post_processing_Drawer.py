import json
import os
import numpy as np
import matplotlib.pyplot as plt


def find_grip_release_indices(all_frames, close_thresh, open_thresh, hold_duration):
    is_gripping = False
    grip_frame_index = -1
    release_frame_index = -1

    potential_grip_start_time = None
    potential_release_start_time = None

    potential_grip_index = -1
    potential_release_index = -1

    for i, frame in enumerate(all_frames):
        timestamp = frame.get("timestamp")
        pinch_dist = frame.get("pinch_m", -1)

        if timestamp is None or pinch_dist < 0:
            potential_grip_start_time = None
            potential_release_start_time = None
            potential_grip_index = -1
            potential_release_index = -1
            continue

        if not is_gripping:
            # GRIP Condition: Pinch is close
            if pinch_dist <= close_thresh:
                if potential_grip_start_time is None:

                    potential_grip_start_time = timestamp
                    potential_grip_index = i
                elif timestamp - potential_grip_start_time >= hold_duration:
                    is_gripping = True

                    grip_frame_index = potential_grip_index
                    potential_grip_start_time = None
                    potential_grip_index = -1
            else:
                # Signal went back up before hold was complete, reset
                potential_grip_start_time = None
                potential_grip_index = -1
        else:
            if pinch_dist >= open_thresh:
                if potential_release_start_time is None:

                    potential_release_start_time = timestamp
                    potential_release_index = i
                elif timestamp - potential_release_start_time >= hold_duration:
                    is_gripping = False

                    release_frame_index = potential_release_index
                    potential_release_start_time = None
                    potential_release_index = -1

            else:
                potential_release_start_time = None
                potential_release_index = -1

    return grip_frame_index, release_frame_index


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
    PINCH_CLOSE_THRESH = 0.05
    PINCH_OPEN_THRESH = 0.12
    HOLD_DURATION = 0.15

    grip_idx, release_idx = find_grip_release_indices(
        all_frames,
        PINCH_CLOSE_THRESH,
        PINCH_OPEN_THRESH,
        HOLD_DURATION
    )

    if grip_idx == -1 or release_idx == -1 or release_idx <= grip_idx:
        print(f"Error: Could not find valid grip/release (grip={grip_idx}, release={release_idx}). Plotting raw data.")
        grip_idx, release_idx = -1, -1
    else:
        print(f"Segmentation found: Grip at frame {grip_idx}, Release at frame {release_idx}")

    timestamps = []
    pinch_distances = []
    for f in all_frames:
        ts = f.get("timestamp")
        pinch = f.get("pinch_m", -1)

        if ts is not None and pinch >= 0:
            timestamps.append(ts)
            pinch_distances.append(pinch)
        else:
            timestamps.append(ts if ts is not None else np.nan)
            pinch_distances.append(np.nan)

    if not timestamps:
        print("No valid timestamp/pinch data to plot.")
        return

    plt.figure(figsize=(15, 5))
    ax = plt.gca()

    ax.plot(timestamps, pinch_distances, label="Pinch Distance (pinch_m)", color='blue', zorder=2)

    ax.axhline(PINCH_CLOSE_THRESH, color='green', linestyle='--', label=f"Close Thresh ({PINCH_CLOSE_THRESH:.2f} m)")
    ax.axhline(PINCH_OPEN_THRESH, color='red', linestyle='--', label=f"Open Thresh ({PINCH_OPEN_THRESH:.2f} m)")

    if grip_idx != -1 and release_idx != -1:
        grip_time = all_frames[grip_idx]['timestamp']
        release_time = all_frames[release_idx]['timestamp']

        ymin, ymax = ax.get_ylim()

        ax.axvline(grip_time, color='green', linestyle='-.', label=f"GRIP (Start Moving)")
        ax.text(grip_time, ymax*0.9, " Reaching", horizontalalignment='right', color='gray')

        ax.axvline(release_time, color='red', linestyle='-.', label=f"RELEASE (Start Retreating)")
        ax.text(release_time, ymax*0.9, " Moving (Opening)", horizontalalignment='left', color='black')
        ax.text(release_time, ymax*0.8, " Retreating", horizontalalignment='right', color='gray')

        # Get valid min/max timestamps for fill
        valid_timestamps = [t for t in timestamps if t is not np.nan]
        min_ts = min(valid_timestamps) if valid_timestamps else 0
        max_ts = max(valid_timestamps) if valid_timestamps else grip_time

        ax.axvspan(min_ts, grip_time, facecolor='gray', alpha=0.1, label="Reaching Phase")
        ax.axvspan(grip_time, release_time, facecolor='green', alpha=0.15, label="Moving Phase")
        ax.axvspan(release_time, max_ts, facecolor='gray', alpha=0.1, label="Retreating Phase")


    ax.set_title("Pinch Distance Segmentation Over Time")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Pinch Distance (meters)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)

    plot_filename = os.path.join(output_dir, "grip_segmentation_plot.png")
    plt.savefig(plot_filename, dpi=150)
    print(f"Successfully saved plot to: {plot_filename}")


if __name__ == "__main__":

    input_file = "/home/ceazar/src/skilllearninglib/data/recordings/3DTest_keypoints3d_drawerWithRetreating1.json"
    output_dir = "/home/ceazar/src/skilllearninglib/data/recordings2/"

    plot_grip_segmentation(input_file, output_dir)

    plt.show()