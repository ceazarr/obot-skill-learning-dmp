import os, json, glob, math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Slerp, Rotation
from cleaning_data import load_demo_json, clean_positions_with_constraints
from helper import (
    smooth_positions, kinematics, forcing_target,
    canonical_phase, pick_joint_index, pick_reference_demo,
    dtw_warp_indices, gmr_predict, calculate_hand_orientation, smooth_orientation_trajectory
)
from orientation_DMP import (
    q_normalize, quaternion_multiply, quaternion_conjugate,
    quaternion_log, q_exp, enforce_hemisphere, derivatives_quat,
)
from visual_gmm import plot_compare_right_wrist_lwr
from ori_visual import plot_orientation_overview

def quats_from_demo(demo):
    t = demo["t"]
    pos = demo["pos"]
    joint_names = demo["joint_names"]
    pos_s = smooth_positions(pos)

    q_xyzw = calculate_hand_orientation(pos_s, joint_names)
    q = np.roll(q_xyzw, shift=1, axis=-1)  # to wxyz
    q = q_normalize(q)
    q = enforce_hemisphere(q)

    return t, q

def prepare_lwr_models(demos, ref_demo, j_idx,
                       alpha_y=25.0, beta_y=None,
                       alpha_z=25.0, beta_z=None,
                       alpha_x=math.log(100.0), eps=1e-8):
    if beta_y is None: beta_y = alpha_y / 4.0
    if beta_z is None: beta_z = alpha_z / 4.0

    Xs_pos, Ys_pos = [], []
    Xs_ori, Ys_ori = [], []
    taus = []

    pos_ref_smoothed = smooth_positions(ref_demo["pos"])
    pos_ref_joint = pos_ref_smoothed[:, j_idx, :]

    for demo in demos:
        t = demo["t"]
        pos_smoothed = smooth_positions(demo["pos"])

        x_demo, tau = canonical_phase(t, alpha_x=alpha_x)
        taus.append(tau)

        V, A = kinematics(pos_smoothed, t)
        g_pos = pos_smoothed[-1]
        y0_pos = pos_smoothed[0]
        f_pos = forcing_target(pos_smoothed, V, A, g_pos, tau, alpha_y, beta_y)

        q_xyzw = calculate_hand_orientation(pos_smoothed, demo["joint_names"])
        q_wxyz = enforce_hemisphere(q_normalize(np.roll(q_xyzw, 1, axis=-1)))

        q_goal = q_wxyz[-1]
        q_init = q_wxyz[0]

        e = 2.0 * quaternion_log(quaternion_multiply(q_goal, quaternion_conjugate(q_wxyz)))
        e0, e1, e2 = derivatives_quat(e, t)

        rhs_ori = (tau**2 * e2) + alpha_z * (beta_z * e0 + tau * e1)
        d = 2.0 * quaternion_log(quaternion_multiply(q_goal, quaternion_conjugate(q_init)))
        D = np.diag(d + 1e-12)
        f_ori = np.linalg.solve(D, rhs_ori.T).T

        pos_demo_joint = pos_smoothed[:, j_idx, :]
        idx_map = dtw_warp_indices(pos_demo_joint, pos_ref_joint)

        x_warp = x_demo[idx_map]
        f_pos_warp = f_pos[idx_map, j_idx, :]
        f_ori_warp = f_ori[idx_map, :]

        mask = (x_warp >= 0.02) & (x_warp <= 0.98)
        if not np.any(mask): continue
        xw = x_warp[mask][:, None]

        k_pos = g_pos[j_idx, :] - y0_pos[j_idx, :]
        k_pos_safe = k_pos + 1e-8
        f_pos_warp_norm = f_pos_warp / k_pos_safe

        yw_pos = f_pos_warp_norm[mask] / (xw + eps)
        good_pos = np.isfinite(yw_pos).all(axis=1)
        if np.any(good_pos):
            Xs_pos.append(xw[good_pos])
            Ys_pos.append(yw_pos[good_pos])

        yw_ori = f_ori_warp[mask] / (xw + eps)
        good_ori = np.isfinite(yw_ori).all(axis=1)
        if np.any(good_ori):
            Xs_ori.append(xw[good_ori])
            Ys_ori.append(yw_ori[good_ori])

    X_pos, Y_pos = np.vstack(Xs_pos), np.vstack(Ys_pos)
    print(f"__| Position LWR model prepared with {len(X_pos)} data points.")

    X_ori, Y_ori = np.vstack(Xs_ori), np.vstack(Ys_ori)
    print(f"__| Orientation LWR model prepared with {len(X_ori)} data points.")

    tau_median = float(np.median(taus))
    pos_model = { "X": X_pos, "Y": Y_pos, "tau": tau_median, "alpha_x": alpha_x, "alpha_y": alpha_y, "beta_y": beta_y }
    ori_model = { "X": X_ori, "Y": Y_ori, "tau": tau_median, "alpha_x": alpha_x, "alpha_z": alpha_z, "beta_z": beta_z }
    return pos_model, ori_model

def lwr_predict(lwr_model, x_query, kernel_width=0.00):
    X_train, Y_train = lwr_model['X'], lwr_model['Y']
    distances = np.abs(X_train - x_query)
    weights = np.exp(-(distances**2) / (2 * kernel_width**2))
    numerator = np.sum(weights * Y_train, axis=0)
    denominator = np.sum(weights, axis=0) + 1e-9
    return numerator / denominator

def rollout_integrated_dmp(pos_model, ori_model, y0, g, q0, qg, T=None, dt=0.01):
    alpha_x = pos_model["alpha_x"]
    tau = pos_model["tau"]
    alpha_y, beta_y = pos_model["alpha_y"], pos_model["beta_y"]
    alpha_z, beta_z = ori_model["alpha_z"], ori_model["beta_z"]

    if T is None: T = tau

    y = np.copy(y0)
    z_pos = np.zeros_like(y)

    k_pos = g - y0
    k_pos_safe = k_pos + 1e-8

    e_ori = 2.0 * quaternion_log(quaternion_multiply(qg, quaternion_conjugate(q0)))
    z_ori = np.zeros_like(e_ori)
    d = 2.0 * quaternion_log(quaternion_multiply(qg, quaternion_conjugate(q0)))
    D = np.diag(d + 1e-12)

    ts = np.arange(0, T + dt, dt)
    position_trajectory = []
    orientation_trajectory = []

    for t in ts:
        x = np.exp(-alpha_x * t / tau)

        f_pos = np.zeros(3)
        f_ori = np.zeros(3)

        if x > 0.01:
            y_shape_pos_norm = lwr_predict(pos_model, x, kernel_width=0.00)
            y_shape_ori = lwr_predict(ori_model, x, kernel_width=0.00)

            f_pos_norm = y_shape_pos_norm * x
            f_ori = y_shape_ori * x
        else:
            f_pos_norm = np.zeros(3)

        f_pos = f_pos_norm * k_pos_safe

        z_dot_pos = (alpha_y * (beta_y * (g - y) - z_pos) + f_pos) / tau
        z_pos += z_dot_pos * dt
        y += (z_pos / tau) * dt
        # --- PERTURBATION TEST ---
        # At t=1.0 seconds, add a sudden 10cm "bump" on the X-axis Y-axis Z-axis
        #if abs(t - 1.0) < (dt / 2.0):
        #    y += np.array([0.1, 0.1, 0.1])

        position_trajectory.append(np.copy(y))

        z_dot_ori = (alpha_z * (beta_z * (-e_ori) - z_ori) + D @ f_ori) / tau
        z_ori += z_dot_ori * dt
        e_ori += (z_ori / tau) * dt

        delta = q_exp(e_ori / 2.0)[0]
        q_current = quaternion_multiply(quaternion_conjugate(delta), qg)
        orientation_trajectory.append(q_normalize(q_current))

    return ts, np.array(position_trajectory), np.array(orientation_trajectory)

def trim_static_start(t, pos, jnames, monitor_joint="Right_Wrist", threshold=0.01):
    try:
        move_joint_idx = jnames.index(monitor_joint)
    except ValueError:
        print(f"Warning: '{monitor_joint}' not found. Using joint 0 for trimming.")
        move_joint_idx = 0

    joint_pos = pos[:, move_joint_idx, :]
    if len(joint_pos) == 0:
        return t, pos

    dist_from_start = np.linalg.norm(joint_pos - joint_pos[0], axis=1)
    start_indices = np.where(dist_from_start > threshold)[0]

    start_idx = 0
    if len(start_indices) > 0:
        start_idx = start_indices[0]
        print(f"Movement detected at index {start_idx}. Trimming {start_idx} static frames.")
    else:
        print(f"Warning: No significant movement detected. Using full demo.")

    t_trimmed = t[start_idx:]
    pos_trimmed = pos[start_idx:, :, :]

    if len(t_trimmed) > 0:
        t_trimmed = t_trimmed - t_trimmed[0]

    return t_trimmed, pos_trimmed

def resample_traj(t_src, y_src, t_dst):
    y_out = np.zeros((len(t_dst), y_src.shape[1]), dtype=float)
    t0, t1 = t_src[0], t_src[-1]
    # Clip to ensure interpolation is within the original time range
    t_clip = np.clip(t_dst, t0, t1)
    for d in range(y_src.shape[1]):
        y_out[:, d] = np.interp(t_clip, t_src, y_src[:, d])
    return y_out
def save_integrated_skill_json(path, meta, pos_model, ori_model, y0, g, q0_wxyz, qg_wxyz, dt=0.01, frame="panda_link0"):

    out = {
        "skill_name": meta.get("skill_name", "lwr_learned_skill"),
        "frame": frame,
        "dt": float(dt),
        "tau": float(pos_model["tau"]),
        "canonical": {"alpha_x": float(pos_model["alpha_x"])},

        "position_dmp": {
            "alpha_y": float(pos_model["alpha_y"]),
            "beta_y":  float(pos_model["beta_y"]),
            "y0": np.asarray(y0, dtype=float).tolist(),
            "g":  np.asarray(g,  dtype=float).tolist(),
            "forcing_function": {
                "type": "LWR",
                "training_data": {
                    "X": pos_model["X"].tolist(),
                    "Y": pos_model["Y"].tolist()
                }
            }
        },

        "orientation_dmp": {
            "alpha_z": float(ori_model["alpha_z"]),
            "beta_z":  float(ori_model["beta_z"]),
            "q0_wxyz": np.asarray(q0_wxyz, dtype=float).tolist(),
            "qg_wxyz": np.asarray(qg_wxyz, dtype=float).tolist(),
            "forcing_function": {
                "type": "LWR",
                "training_data": {
                    "X": ori_model["X"].tolist(),
                    "Y": ori_model["Y"].tolist()
                }
            }
        },

        "meta": meta
    }

    with open(path, "w") as f:
        json.dump(out, f, indent=4)
    print(f"\n[save] Integrated skill with LWR models saved to: {path}")
def joint_traj_3d(demo, j_idx):
    P = smooth_positions(demo["pos"])
    return P[:, j_idx, :]

def plot_demos_and_generated_3d(ax, demos, j_idx, ref_smt, gen, title):
    for d in demos:
        traj = joint_traj_3d(d, j_idx)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                linewidth=1.0, alpha=0.35, color="#BBBBBB")

    ax.plot(ref_smt[:, 0], ref_smt[:, 1], ref_smt[:, 2],
            linewidth=2.0, alpha=0.9, color="#666666", label="smooth ref")

    ax.plot(gen[:, 0], gen[:, 1], gen[:, 2],
            linestyle="--", linewidth=2.0, label="Generated")

    ax.scatter(ref_smt[0, 0],  ref_smt[0, 1],  ref_smt[0, 2],  s=50, label="start")
    ax.scatter(ref_smt[-1, 0], ref_smt[-1, 1], ref_smt[-1, 2], s=50, label="end")

    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(loc="best")

if __name__ == "__main__":
    input_glob = r"/home/ceazar/src/skilllearninglib/data/recordings/movingNew3.json"
    files = sorted(glob.glob(input_glob))
    if not files:
        raise ValueError("No files found matching the pattern.")
    print(f"Found {len(files)} demonstration files.")

    demos = []
    joint_names_ref = None
    for p in files:
        print(f"--- Processing: {os.path.basename(p)} ---")
        t, pos_raw, jnames = load_demo_json(p)
        t_clean, pos_clean = clean_positions_with_constraints(t, pos_raw, jnames)
        t_trimmed, pos_trimmed = trim_static_start(
            t_clean, pos_clean, jnames, monitor_joint="Right_Wrist", threshold=0.01
        )
        if len(t_trimmed) < 10:
            print(f"Skipping demo {os.path.basename(p)}: too short after trimming.")
            continue
        if joint_names_ref is None:
            joint_names_ref = jnames
        demos.append({
            "file": os.path.basename(p),
            "t": t_trimmed,
            "pos": pos_trimmed,
            "joint_names": jnames
        })
    print(f"\n[load] Finished. Loaded, cleaned, and trimmed {len(demos)} demos successfully.")

    j_idx, j_name = pick_joint_index(joint_names_ref, "Right_Wrist")
    ref_demo = pick_reference_demo(demos, j_idx)
    print(f"Using '{j_name}' as the end-effector.")
    print(f"Using '{ref_demo['file']}' as the DTW reference.")

    pos_model, ori_model = prepare_lwr_models(
        demos, ref_demo, j_idx,
        alpha_y=25.0, alpha_z=25.0, alpha_x=math.log(100.0)
    )

    pos_ref_smoothed = smooth_positions(ref_demo['pos'])
    y0, g = pos_ref_smoothed[0, j_idx, :], pos_ref_smoothed[-1, j_idx, :]
    _, q_wxyz_ref = quats_from_demo(ref_demo)
    q0, qg = q_wxyz_ref[0], q_wxyz_ref[-1]
    meta = {
        "skill_name": "reach_like_demo",
        "files": [d["file"] for d in demos],
        "ref_demo": ref_demo["file"],
        "ee_joint": j_name,
        "notes": "Integrated pos+ori DMP with shared canonical phase"
    }
    save_integrated_skill_json(
        "two_dmps_lwr.json", meta,
        pos_model, ori_model,
        y0, g, q0, qg,
        dt=0.01, frame="panda_link0",
    )

    print("\n--- Starting Rollout ---")

    t_rollout, P_rollout, Q_rollout = rollout_integrated_dmp(pos_model, ori_model, y0, g, q0, qg)
    print(f"Generated {len(t_rollout)} synchronized 6-DoF trajectory points.")

    print("\n--- Visualizing Comparison ---")

    raw_ref_j = ref_demo["pos"][:, j_idx, :]
    smt_ref_j = pos_ref_smoothed[:, j_idx, :]

    raw_ref_rs = resample_traj(ref_demo["t"], raw_ref_j, t_rollout)
    smt_ref_rs = resample_traj(ref_demo["t"], smt_ref_j, t_rollout)

    plot_compare_right_wrist_lwr(
        t_rollout, raw_ref_rs, smt_ref_rs, P_rollout,
        title="Integrated Position DMP vs. Reference (LWR)",
        show=False
    )
    fig_pos3d = plt.figure(figsize=(9, 9))
    ax3d = fig_pos3d.add_subplot(1, 1, 1, projection="3d")

    ref_smt = smt_ref_rs
    plot_demos_and_generated_3d(
        ax3d,
        demos=demos,
        j_idx=j_idx,
        ref_smt=ref_smt,
        gen=P_rollout,
        title="Position (EE) with Training Demos (light grey) + Generated"
    )
    try:
        ax3d.set_box_aspect([1, 1, 1])  # needs newer matplotlib
    except Exception:
        pass
    fig2d, axs = plt.subplots(1, 3, figsize=(14, 4))

    # X–Y
    axs[0].plot(raw_ref_rs[:, 0], raw_ref_rs[:, 1], label="ref (raw)", alpha=0.3)
    axs[0].plot(smt_ref_rs[:, 0], smt_ref_rs[:, 1], label="ref (smooth)", linewidth=2)
    axs[0].plot(P_rollout[:, 0], P_rollout[:, 1], "--", label="generated", linewidth=2)
    axs[0].set_xlabel("X"); axs[0].set_ylabel("Y"); axs[0].set_title("XY")
    axs[0].legend()

    # X–Z
    axs[1].plot(raw_ref_rs[:, 0], raw_ref_rs[:, 2], label="ref (raw)", alpha=0.3)
    axs[1].plot(smt_ref_rs[:, 0], smt_ref_rs[:, 2], label="ref (smooth)", linewidth=2)
    axs[1].plot(P_rollout[:, 0], P_rollout[:, 2], "--", label="generated", linewidth=2)
    axs[1].set_xlabel("X"); axs[1].set_ylabel("Z"); axs[1].set_title("XZ")

    # Y–Z
    axs[2].plot(raw_ref_rs[:, 1], raw_ref_rs[:, 2], label="ref (raw)", alpha=0.3)
    axs[2].plot(smt_ref_rs[:, 1], smt_ref_rs[:, 2], label="ref (smooth)", linewidth=2)
    axs[2].plot(P_rollout[:, 1], P_rollout[:, 2], "--", label="generated", linewidth=2)
    axs[2].set_xlabel("Y"); axs[2].set_ylabel("Z"); axs[2].set_title("YZ")

    plt.tight_layout()

    t_ref_o, q_ref_wxyz = quats_from_demo(ref_demo)

    idx = np.searchsorted(t_ref_o, np.clip(t_rollout, t_ref_o[0], t_ref_o[-1]))
    idx = np.clip(idx, 0, len(q_ref_wxyz) - 1)
    q_ref_match = q_ref_wxyz[idx]

    #plot_orientation_overview(
    #    t_rollout, q_ref_match, Q_rollout,
    #    alpha_x=pos_model["alpha_x"],
    #    title="Integrated Orientation DMP vs. Reference (LWR)"
    #)
    # --- New Euler Angle Plot (Yaw, Pitch, Roll) ---
    print("Generating Euler angle (YPR) plot...")

    q_ref_xyzw = np.roll(q_ref_match, shift=-1, axis=-1)
    q_gen_xyzw = np.roll(Q_rollout, shift=-1, axis=-1)

    r_ref = Rotation.from_quat(q_ref_xyzw)
    r_gen = Rotation.from_quat(q_gen_xyzw)

    euler_ref = r_ref.as_euler('zyx')
    euler_gen = r_gen.as_euler('zyx')


    euler_ref_unwrapped = np.unwrap(euler_ref, axis=0)
    euler_gen_unwrapped = np.unwrap(euler_gen, axis=0)

    fig_euler, ax_euler = plt.subplots(figsize=(12, 6))

    colors = ['C0', 'C1', 'C2']
    labels = ['Yaw (Z)', 'Pitch (Y)', 'Roll (X)']

    for i in range(3):
        ax_euler.plot(t_rollout, euler_ref_unwrapped[:, i],
                      label=f'Reference {labels[i]}', color=colors[i], linestyle='--')

        ax_euler.plot(t_rollout, euler_gen_unwrapped[:, i],
                      label=f'Generated {labels[i]}', color=colors[i], linestyle='-')

    ax_euler.set_title('Orientation (Yaw, Pitch, Roll) Comparison')
    ax_euler.set_xlabel('Time (s)')
    ax_euler.set_ylabel('rad')
    ax_euler.legend(loc='best')
    ax_euler.grid(True)
    plt.tight_layout()

    print("\n--- Trajectory Error Evaluation ---")


    min_len = min(len(smt_ref_rs), len(P_rollout), len(q_ref_match), len(Q_rollout))

    ref_pos_traj = smt_ref_rs[:min_len]
    gen_pos_traj = P_rollout[:min_len]

    squared_error_per_axis = np.square(ref_pos_traj - gen_pos_traj)

    mse_per_axis = np.mean(squared_error_per_axis, axis=0)
    print(f"[Position] MSE (X, Y, Z): [{mse_per_axis[0]:.6f}, {mse_per_axis[1]:.6f}, {mse_per_axis[2]:.6f}] m^2")

    squared_euclidean_dist = np.sum(squared_error_per_axis, axis=1)
    total_mse_pos = np.mean(squared_euclidean_dist)

    rmse_pos = np.sqrt(total_mse_pos)
    print(f"[Position] MSE (Euclidean Total): {total_mse_pos:.6f} m^2")
    print(f"[Position] RMSE (Euclidean Total): {rmse_pos:.6f} m")


    ref_ori_traj = q_ref_match[:min_len]
    gen_ori_traj = Q_rollout[:min_len]

    dot_products = np.sum(gen_ori_traj * ref_ori_traj, axis=1)
    ref_ori_compare = np.copy(ref_ori_traj)
    mask = dot_products < 0
    ref_ori_compare[mask] = -ref_ori_compare[mask]

    q_diff = quaternion_multiply(gen_ori_traj, quaternion_conjugate(ref_ori_compare))


    w_clamped = np.clip(q_diff[:, 0], -1.0, 1.0)
    angles_rad = 2.0 * np.arccos(w_clamped)

    ori_squared_error_rad = np.square(angles_rad)
    mse_orientation_rad = np.mean(ori_squared_error_rad)
    rmse_orientation_rad = np.sqrt(mse_orientation_rad)
    # Convert to degrees for easier interpretation
    rmse_orientation_deg = rmse_orientation_rad * (180.0 / np.pi)

    print(f"\n[Orientation] MSE (Angle): {mse_orientation_rad:.6f} (rad^2)")
    print(f"[Orientation] RMSE (Angle): {rmse_orientation_deg:.4f} (degrees)")
    print("-----------------------------------------------\n")
    plt.show()