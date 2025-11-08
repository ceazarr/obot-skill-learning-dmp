import os, glob, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.widgets import Slider
from helper import (smooth_positions, pick_joint_index,
                    canonical_phase, dtw_warp_indices, pick_reference_demo, calculate_hand_orientation, smooth_orientation_trajectory)
from cleaning_data import interpolate_nan_values, load_demo_json, clean_positions_with_constraints
from scipy.spatial.transform import Rotation as R
from orientation_DMP import (q_normalize, quaternion_multiply, quaternion_conjugate)

def smooth_position_visualize(t, raw, smt, title="Position smoothing", show=True, block=None):
    fig = plt.figure(figsize=(16, 5))
    labels = ["X","Y","Z"]
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1)
        ax.plot(t, raw[:, i], alpha=0.5, label="raw")
        ax.plot(t, smt[:, i], alpha=0.9, label="smoothed")
        ax.set_title(f"{title} – {labels[i]}(t)")
        ax.set_xlabel("time (s)"); ax.set_ylabel(labels[i])
        ax.legend(loc="best")
    plt.tight_layout()
    if show:
        plt.show(block=block)
    return fig


def plot_compare_right_wrist(t, raw, smt, gen, title="Right_Wrist comparison", show=True, block=None):
    fig = plt.figure(figsize=(16, 10))

    labels = ["X","Y","Z"]
    for i in range(3):
        ax = fig.add_subplot(2, 3, i+1)
        ax.plot(t, raw[:, i], alpha=0.5, label="raw")
        ax.plot(t, smt[:, i], alpha=0.9, label="smoothed")
        ax.plot(t, gen[:, i], "--", linewidth=2, label="GMR+DMP")
        ax.set_title(f"{title} – {labels[i]}(t)")
        ax.set_xlabel("time (s)"); ax.set_ylabel(labels[i])
        ax.legend(loc="best")

    ax3d = fig.add_subplot(2, 3, 4, projection="3d")
    ax3d.plot(raw[:,0], raw[:,1], raw[:,2], alpha=0.5, label="raw")
    ax3d.plot(smt[:,0], smt[:,1], smt[:,2], alpha=0.9, label="smoothed")
    ax3d.plot(gen[:,0], gen[:,1], gen[:,2], "--", linewidth=2, label="GMR+DMP")
    ax3d.scatter(raw[0,0], raw[0,1], raw[0,2], s=50, label="start")
    ax3d.scatter(raw[-1,0], raw[-1,1], raw[-1,2], s=50, label="end")
    ax3d.set_title(f"{title} – 3D path")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.legend()

    # Position MSE over time (smoothed vs generated)
    ax_err = fig.add_subplot(2, 3, 5)
    err_t = np.mean((smt - gen)**2, axis=1)
    ax_err.plot(t, err_t)
    ax_err.set_title("Position MSE (smoothed vs GMR+DMP) over time")
    ax_err.set_xlabel("time (s)"); ax_err.set_ylabel("MSE")

    plt.tight_layout()
    if show:
        plt.show()
    return fig

def plot_compare_right_wrist_lwr(t, raw, smt, gen, title="Right_Wrist comparison", show=True, block=None):
    fig = plt.figure(figsize=(16, 10))

    labels = ["X","Y","Z"]
    for i in range(3):
        ax = fig.add_subplot(2, 3, i+1)
        ax.plot(t, raw[:, i], alpha=0.5, label="raw")
        ax.plot(t, smt[:, i], alpha=0.9, label="smoothed")
        ax.plot(t, gen[:, i], "--", linewidth=2, label="LR+DMP")
        ax.set_title(f"{title} – {labels[i]}(t)")
        ax.set_xlabel("time (s)"); ax.set_ylabel(labels[i])
        ax.legend(loc="best")

    ax3d = fig.add_subplot(2, 3, 4, projection="3d")
    ax3d.plot(raw[:,0], raw[:,1], raw[:,2], alpha=0.5, label="raw")
    ax3d.plot(smt[:,0], smt[:,1], smt[:,2], alpha=0.9, label="smoothed")
    ax3d.plot(gen[:,0], gen[:,1], gen[:,2], "--", linewidth=2, label="LR+DMP")
    ax3d.scatter(raw[0,0], raw[0,1], raw[0,2], s=50, label="start")
    ax3d.scatter(raw[-1,0], raw[-1,1], raw[-1,2], s=50, label="end")
    ax3d.set_title(f"{title} – 3D path")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.legend()

    ax_err = fig.add_subplot(2, 3, 5)
    err_t = np.mean((smt - gen)**2, axis=1)
    ax_err.plot(t, err_t)
    ax_err.set_title("Position MSE (smoothed vs GMR+DMP) over time")
    ax_err.set_xlabel("time (s)"); ax_err.set_ylabel("MSE")

    plt.tight_layout()
    if show:
        plt.show()
    return fig

#def load_all_demos(input_glob):
#    files = sorted(glob.glob(input_glob))
#    if not files:
#        raise FileNotFoundError(f"No files match: {input_glob}")
#    demos = []
#    joint_names_ref = None
#    for p in files:
#        t, pos, jnames = load_demo_json(p)
#        t, pos_clean = clean_positions_with_constraints(t, pos, jnames)
#        if joint_names_ref is None:
#            joint_names_ref = jnames
#        else:
#            if jnames != joint_names_ref:
#                raise ValueError(f"Joint order mismatch in {p}")
#        demos.append({"file": os.path.basename(p), "t": t, "pos": pos_clean})
#    return demos, joint_names_ref
POSSIBLE_KEYS = [
    "keypoints_in_cup_frame",
    "keypoints_in_drawer_frame_current",
    "keypoints_in_drawer_frame",
    "drawer0_frame",
    "table_frame"
]

def load_all_demos(path):
    import json, numpy as np
    with open(path, "r") as f:
        frames = json.load(f)
    if not frames:
        raise ValueError("Empty recording.")

    found = None
    for k in POSSIBLE_KEYS:
        if k in frames[0]:
            found = k
            break
    if found is None:
        raise KeyError(f"None of the expected keys {POSSIBLE_KEYS} found in first frame.")

    joint_names = list(frames[0][found].keys())

    t = np.array([fr["timestamp"] for fr in frames], dtype=float)
    pos = np.array([[fr[found][jn] for jn in joint_names] for fr in frames], dtype=float)  # shape (T, J, 3)
    return t, pos, joint_names

def overlay_joint_demos(input_glob,
                        focus_joint="Right_Wrist",
                        align="phase",
                        smooth=True,
                        normalize="none",
                        x_min_phase=0.0,
                        show=True, block=None):
    demos, joint_names = load_all_demos(input_glob)
    j_idx, j_name = pick_joint_index(joint_names, focus_joint)

    ref_demo = pick_reference_demo(demos, j_idx)
    P_ref = smooth_positions(ref_demo["pos"])
    p_ref = P_ref[:, j_idx, :]

    fig = plt.figure(figsize=(16, 10))
    axX = fig.add_subplot(2, 3, 1)
    axY = fig.add_subplot(2, 3, 2)
    axZ = fig.add_subplot(2, 3, 3)
    ax3d = fig.add_subplot(2, 3, 4, projection="3d")
    axXY = fig.add_subplot(2, 3, 5)

    def _label(d): return os.path.splitext(d["file"])[0]
    xlabel = "time (s)" if align == "time" else ("canonical phase x" if align == "phase" else "DTW index (ref grid)")

    for d in demos:
        t, pos = d["t"], d["pos"]
        P = smooth_positions(pos) if smooth else pos
        p = P[:, j_idx, :]  # (T,3)

        if align == "time":
            x = t - t[0]
            p_base = p
        elif align == "phase":
            x, _ = canonical_phase(t, alpha_x=math.log(100.0))
            mask = x >= x_min_phase
            x = x[mask]
            p_base = p[mask]
        elif align == "dtw":
            idx_map = dtw_warp_indices(p, p_ref)  # (T_ref,)
            p_base = p[idx_map]                    # aligned to ref lengt
            x = np.arange(len(p_base))
        else:
            raise ValueError("align must be 'time', 'phase', or 'dtw'")

        y0 = p_base[0].copy()
        g  = p_base[-1].copy()
        A  = g - y0
        A[np.abs(A) < 1e-6] = 1.0

        if normalize == "none":
            p_plot = p_base
        elif normalize == "center":
            p_plot = p_base - y0
        elif normalize == "amp":
            p_plot = (p_base - y0) / A
        else:
            raise ValueError("normalize must be 'none', 'center', or 'amp'")

        # 1D plots
        axX.plot(x, p_plot[:, 0], label=_label(d), alpha=0.9)
        axY.plot(x, p_plot[:, 1], label=_label(d), alpha=0.9)
        axZ.plot(x, p_plot[:, 2], label=_label(d), alpha=0.9)
        # 3D + XY
        ax3d.plot(p_plot[:, 0], p_plot[:, 1], p_plot[:, 2], alpha=0.9)
        ax3d.scatter(p_plot[0,0], p_plot[0,1], p_plot[0,2], s=30)
        ax3d.scatter(p_plot[-1,0], p_plot[-1,1], p_plot[-1,2], s=30)
        axXY.plot(p_plot[:, 0], p_plot[:, 1], alpha=0.9)

    norm_tag = {"none":"raw", "center":"centered", "amp":"amplitude-norm"}[normalize]
    title_tag = f"{j_name} — align:{align} | {norm_tag} | smooth:{smooth}"

    for ax, lab in zip((axX, axY, axZ), ("X","Y","Z")):
        ax.set_title(f"{lab} vs {xlabel}"); ax.set_xlabel(xlabel); ax.set_ylabel(lab)
    axX.legend(fontsize=8, ncols=2, loc="best")

    ax3d.set_title(f"{j_name} 3D overlay"); ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    axXY.set_title(f"{j_name} XY projection"); axXY.set_xlabel("X"); axXY.set_ylabel("Y")

    fig.suptitle(title_tag, fontsize=14)
    plt.tight_layout()
    if show:
        plt.show(block=block)
    return fig

def plot_quaternion_components(t, quats, title="Quaternion Components Over Time", show=True, block=True):

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = ['x', 'y', 'z', 'w']
    for i in range(4):
        ax.plot(t, quats[:, i], label=labels[i])

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Quaternion Value")
    ax.legend()
    ax.grid(True, linestyle='--')
    plt.tight_layout()

    if show:
        plt.show(block=block)

    return fig

def plot_orientation_trajectory(t, pos_clean, joint_names, title="Hand Orientation Trajectory", show=True, block=True):

    print("Calculating orientation for visualization...")
    quats = calculate_hand_orientation(pos_clean, joint_names)
    wrist_idx = joint_names.index("Right_Wrist")
    thumb_idx = joint_names.index("Right_Thumb_Tip")
    index_idx = joint_names.index("Right_Index_Tip")

    wrist_traj = pos_clean[:, wrist_idx, :]
    thumb_traj = pos_clean[:, thumb_idx, :]
    index_traj = pos_clean[:, index_idx, :]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.1, 0.2, 0.8, 0.75], projection='3d')
    fig.suptitle(title, fontsize=16)

    min_coords = np.min(wrist_traj, axis=0); max_coords = np.max(wrist_traj, axis=0)
    center = (max_coords + min_coords) / 2
    max_range = np.max(max_coords - min_coords) * 0.7

    ax.plot(wrist_traj[:, 0], wrist_traj[:, 1], wrist_traj[:, 2], color='gray', alpha=0.5, linestyle=':', label='Wrist Path')

    wrist_point, = ax.plot([], [], [], 'ko', markersize=8, label="Wrist")
    thumb_point, = ax.plot([], [], [], 'co', markersize=5, label="Thumb Tip")
    index_point, = ax.plot([], [], [], 'mo', markersize=5, label="Index Tip")

    axis_len = 0.08
    rot_matrix_init = R.from_quat(quats[0]).as_matrix()
    x_axis_vec = rot_matrix_init[:, 0] * axis_len; y_axis_vec = rot_matrix_init[:, 1] * axis_len; z_axis_vec = rot_matrix_init[:, 2] * axis_len

    x_axis = ax.quiver(wrist_traj[0,0], wrist_traj[0,1], wrist_traj[0,2], x_axis_vec[0], x_axis_vec[1], x_axis_vec[2], color='r', label='Hand X')
    y_axis = ax.quiver(wrist_traj[0,0], wrist_traj[0,1], wrist_traj[0,2], y_axis_vec[0], y_axis_vec[1], y_axis_vec[2], color='g', label='Hand Y')
    z_axis = ax.quiver(wrist_traj[0,0], wrist_traj[0,1], wrist_traj[0,2], z_axis_vec[0], z_axis_vec[1], z_axis_vec[2], color='b', label='Hand Z')

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    ax.legend()
    ax.view_init(elev=30, azim=60)

    ax_slider = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    time_slider = Slider(ax=ax_slider, label='Time (s)', valmin=t[0], valmax=t[-1], valinit=t[0], valstep=(t[1]-t[0]))

    def update(val):
        idx = np.argmin(np.abs(t - time_slider.val))

        wrist_point.set_data_3d([wrist_traj[idx, 0]], [wrist_traj[idx, 1]], [wrist_traj[idx, 2]])
        thumb_point.set_data_3d([thumb_traj[idx, 0]], [thumb_traj[idx, 1]], [thumb_traj[idx, 2]])
        index_point.set_data_3d([index_traj[idx, 0]], [index_traj[idx, 1]], [index_traj[idx, 2]])

        origin = wrist_traj[idx]
        rot_matrix = R.from_quat(quats[idx]).as_matrix()

        nonlocal x_axis, y_axis, z_axis
        x_axis.remove(); y_axis.remove(); z_axis.remove()

        x_vec = rot_matrix[:, 0] * axis_len; y_vec = rot_matrix[:, 1] * axis_len; z_vec = rot_matrix[:, 2] * axis_len

        x_axis = ax.quiver(origin[0], origin[1], origin[2], x_vec[0], x_vec[1], x_vec[2], color='r')
        y_axis = ax.quiver(origin[0], origin[1], origin[2], y_vec[0], y_vec[1], y_vec[2], color='g')
        z_axis = ax.quiver(origin[0], origin[1], origin[2], z_vec[0], z_vec[1], z_vec[2], color='b')

        fig.canvas.draw_idle()

    time_slider.on_changed(update)

    if show:
        plt.show(block=block)

    return fig, time_slider


def geodesic_angle(q, qg):

    q = np.atleast_2d(q); qg = np.atleast_2d(qg)
    dq = quaternion_multiply(q, quaternion_conjugate(qg))
    dq = q_normalize(dq)
    w = np.clip(dq[:, 0], -1.0, 1.0)
    v = dq[:, 1:]
    theta = 2.0 * np.arctan2(np.linalg.norm(v, axis=1), abs(w))
    return theta

def plot_orientation_comparison(t, q_demo_wxyz, q_gen_wxyz, title="Orientation: demo vs DMP", show=True):
    # Components
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(2,1,1)
    labels = ['w','x','y','z']
    for i, lab in enumerate(labels):
        ax1.plot(t, q_demo_wxyz[:, i], alpha=0.5, label=f"demo {lab}")
        ax1.plot(t, q_gen_wxyz[:, i],  linestyle="--", label=f"DMP {lab}")
    ax1.set_title(title + " (components)")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Quaternion components")
    ax1.legend(ncols=4); ax1.grid(True, linestyle=":")

    # Geodesic angle error to demo
    ax2 = fig.add_subplot(2,1,2)
    theta = geodesic_angle(q_gen_wxyz, q_demo_wxyz)
    ax2.plot(t, np.degrees(theta))
    ax2.set_title("Geodesic error to demo (deg)")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("deg")
    ax2.grid(True, linestyle=":")

    plt.tight_layout()
    if show: plt.show()
    return fig

def plot_orientation_components_split(t, q_demo_wxyz, q_dmp_wxyz,
                                      title="Orientation vs ref (split components)",
                                      show=True, block=True):
    q_demo_wxyz = np.asarray(q_demo_wxyz)
    q_dmp_wxyz  = np.asarray(q_dmp_wxyz)
    assert q_demo_wxyz.shape == q_dmp_wxyz.shape and q_demo_wxyz.shape[1] == 4

    labels = ['w', 'x', 'y', 'z']
    styles_demo = dict(alpha=0.9, linewidth=1.8)
    styles_dmp  = dict(alpha=0.9, linewidth=2.0, linestyle='--')

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for i, lab in enumerate(labels):
        ax = axes[i]
        ax.plot(t, q_demo_wxyz[:, i], label=f"demo {lab}", **styles_demo)
        ax.plot(t, q_dmp_wxyz[:, i],  label=f"DMP {lab}",  **styles_dmp)
        ax.set_ylabel(lab)
        ax.grid(True, linestyle=":")
        ax.legend(loc="best", ncols=2, fontsize=9)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if show:
        plt.show(block=block)
    return fig


if __name__ == "__main__":

    input_file = "/data/recordings3/drawer_retreating1.json"
    print(f"Loading and cleaning data from {input_file}...")
    t, pos_raw, joint_names = load_demo_json(input_file)
    t, pos_clean = clean_positions_with_constraints(t, pos_raw, joint_names)
    print("Data successfully cleaned.")

    if all(j in joint_names for j in ["Right_Wrist", "Right_Thumb_Tip", "Right_Index_Tip"]):
        quaternions = calculate_hand_orientation(pos_clean, joint_names)

        plot_quaternion_components(t, quaternions, block=True)

        plot_orientation_trajectory(t, pos_clean, joint_names, block=True)
    else:
        print("Error: The required joints (Right_Wrist, Right_Thumb_Tip, Right_Index_Tip) were not found.")

