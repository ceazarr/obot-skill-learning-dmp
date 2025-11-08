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
from visual_gmm import plot_compare_right_wrist
from ori_visual import plot_orientation_overview

def quats_from_demo(demo):
    t = demo["t"]
    pos = demo["pos"]
    joint_names = demo["joint_names"]
    pos_s = smooth_positions(pos)

    q_xyzw = smooth_orientation_trajectory(pos_s, joint_names)
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
        f_pos = forcing_target(pos_smoothed, V, A, g_pos, tau, alpha_y, beta_y)

        q_xyzw = smooth_orientation_trajectory(pos_smoothed, demo["joint_names"])
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

        yw_pos = f_pos_warp[mask] / (xw + eps)
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

def lwr_predict(lwr_model, x_query, kernel_width=0.04):
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
    
    e_ori = 2.0 * quaternion_log(quaternion_multiply(qg, quaternion_conjugate(q0)))
    z_ori = np.zeros_like(e_ori)
    d = 2.0 * quaternion_log(quaternion_multiply(qg, quaternion_conjugate(q0)))
    D = np.diag(d + 1e-12)

    #timesteps = int(T / dt) 
    #ts = np.linspace(0, T, timesteps)
    ts = np.arange(0, T + dt, dt)
    position_trajectory = []
    orientation_trajectory = []

    for t in ts:
        x = np.exp(-alpha_x * t / tau)
        
        f_pos = np.zeros(3)
        f_ori = np.zeros(3)

        if x > 0.01: 
            y_shape_pos = lwr_predict(pos_model, x, kernel_width=0.04)
            y_shape_ori = lwr_predict(ori_model, x, kernel_width=0.02)
            f_pos = y_shape_pos * x
            f_ori = y_shape_ori * x
        
        z_dot_pos = (alpha_y * (beta_y * (g - y) - z_pos) + f_pos) / tau
        z_pos += z_dot_pos * dt
        y += (z_pos / tau) * dt
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
    
if __name__ == "__main__":
    input_glob = r"C:/Users/ceaz/OneDrive/Desktop/Code for thesis/recordings2/*NewOri*.json"
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
        alpha_y=25.0, alpha_z=80.0, alpha_x=math.log(100.0)
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
    
    plot_compare_right_wrist(
        t_rollout, raw_ref_rs, smt_ref_rs, P_rollout,
        title="Integrated Position DMP vs. Reference (LWR)",
        show=False
    )
    
    t_ref_o, q_ref_wxyz = quats_from_demo(ref_demo)
    
    idx = np.searchsorted(t_ref_o, np.clip(t_rollout, t_ref_o[0], t_ref_o[-1]))
    idx = np.clip(idx, 0, len(q_ref_wxyz) - 1)
    q_ref_match = q_ref_wxyz[idx]

    plot_orientation_overview(
        t_rollout, q_ref_match, Q_rollout,
        alpha_x=pos_model["alpha_x"],
        title="Integrated Orientation DMP vs. Reference (LWR)"
    )

    plt.show()