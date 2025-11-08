import os, glob, json, math
import numpy as np
from scipy import ndimage
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from gmr import GMM as GMR
from helper import (smooth_positions, canonical_phase,
                    pick_joint_index, dtw_warp_indices,
                    fit_gmm_bic, gmr_predict, pick_reference_demo, calculate_hand_orientation
                    )
from visual_gmm import (plot_quaternion_components,
                        plot_orientation_comparison, plot_orientation_components_split)
from cleaning_data import load_demo_json, interpolate_nan_values, clean_positions_with_constraints
from orientation_DMP import (q_normalize, quaternion_multiply, quaternion_log, q_exp, enforce_hemisphere,
                             quaternion_conjugate, derivatives_quat)
from io_model import save_orientation_model
from ori_visual import plot_orientation_overview, overlay_orientation_demos, plot_axis_trajectory_on_sphere, plot_axis_trajectories_on_sphere_overlay

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

def build_training_pairs_for_orientation(demox, alpha_z=20.0, beta_z=None, alpha_x=math.log(100.0), eps=1e-8):
    if beta_z is None:
        beta_z = alpha_z / 4.0

    Xs = []
    Ys = []
    taus = []

    for demo in demox:
        t, q_seq = quats_from_demo(demo)
        q_goal = q_seq[-1]
        tau = max(t[-1] - t[0], 1e-12)
        taus.append(tau)

        e = quaternion_log(quaternion_multiply(q_goal, quaternion_conjugate(q_seq))) * 2.0

        e0, e1, e2 = derivatives_quat(e, t, window_length=9, polyorder=3)
        rhs = (tau**2) * e2 + alpha_z * (beta_z * e0 + tau * e1)

        d = 2.0 * quaternion_log(quaternion_multiply(q_goal, quaternion_conjugate(q_seq[0])))
        D = np.diag(d + 1e-12)
        f = np.linalg.solve(D, rhs.T).T

        x, _ = canonical_phase(t, alpha_x=alpha_x)
        y = f / (x[:, None] + eps)
        mask = (x >= 0.02) & (x <= 0.98)
        xw = x[mask][:, None]
        yw = (f / (x[:, None] + eps))[mask]
        Xs.append(xw); Ys.append(yw)

    X = np.vstack(Xs)  # (sum T, 1)
    Y = np.vstack(Ys)  # (sum T, 3)
    tau_median = float(np.median(taus))

    return X, Y, tau_median, dict(alpha_z=alpha_z, beta_z=beta_z, alpha_x=alpha_x)

def build_training_pairs_for_orientation_dtw(demos, ref_demo, alpha_z=20.0, beta_z=None, alpha_x=math.log(100.0), eps=1e-8):
    if beta_z is None: beta_z = alpha_z / 4.0

    t_ref, q_ref = quats_from_demo(ref_demo)
    q_goal_ref = q_ref[-1]
    e_ref = quaternion_log(quaternion_multiply(q_goal_ref, quaternion_conjugate(q_ref))) * 2.0
    Xs, Ys, taus = [], [], []
    for demo in demos:
        t, q_seq = quats_from_demo(demo)
        q_goal = q_seq[-1]
        tau = max(t[-1] - t[0], 1e-12); taus.append(tau)

        e  = quaternion_log(quaternion_multiply(q_goal, quaternion_conjugate(q_seq))) * 2.0
        e0, e1, e2 = derivatives_quat(e, t, window_length=11, polyorder=3)
        rhs = (tau**2) * e2 + alpha_z * (beta_z * e0 + tau * e1)

        d = 2.0 * quaternion_log(quaternion_multiply(q_goal, quaternion_conjugate(q_seq[0])))
        D = np.diag(d + 1e-12)
        f = np.linalg.solve(D, rhs.T).T
        
        x_demo, _ = canonical_phase(t, alpha_x=alpha_x)

        idx_map = dtw_warp_indices(e, e_ref)
        f_warp  = f[idx_map, :]
        x_warp  = x_demo[idx_map]

        # ---- NUMERICAL SAFEGUARDS ----
        mask = (x_warp >= 0.02) & (x_warp <= 0.98)   # drop extremes where /x explodes
        if not np.any(mask):
            continue
        xw = x_warp[mask][:, None]
        fw = f_warp[mask]
        y = fw / (xw + eps)

        good = np.isfinite(y).all(axis=1) & np.isfinite(xw).all(axis=1)
        if np.any(good):
            Xs.append(xw[good])
            Ys.append(y[good])

    if not Xs:
        raise ValueError("No valid training pairs after DTW+masking; check input demos.")

    X = np.vstack(Xs)
    Y = np.vstack(Ys)
    tau_median = float(np.median(taus))
    return X, Y, tau_median, dict(alpha_z=alpha_z, beta_z=beta_z, alpha_x=alpha_x)



def fit_orientation_dmp_gmm(demos, use_dtw=True, ref_demo=None, alpha_z=20.0, beta_z=None, alpha_x=math.log(100.0)):
    if use_dtw:
        assert ref_demo is not None, "Provide ref_demo when use_dtw=True"
        X, Y, tau_med, params = build_training_pairs_for_orientation_dtw(
            demos, ref_demo, alpha_z=alpha_z, beta_z=beta_z, alpha_x=alpha_x
        )
    else:
        X, Y, tau_med, params = build_training_pairs_for_orientation(
            demos, alpha_z=alpha_z, beta_z=beta_z, alpha_x=alpha_x
        )

    Z = np.hstack([X, Y])  # shape (N, 4)

    # --- sanitize before GMM ---
    finite = np.isfinite(Z).all(axis=1)
    Z = Z[finite]
    if Z.shape[0] < 5:
        raise ValueError(f"Too few valid samples for GMM after filtering: N={Z.shape[0]}")
    # prevent Ks > N-1
    #max_K = max(2, min(16, Z.shape[0] - 1))

    gmm = fit_gmm_bic(Z, Ks=range(2, 50), n_init=12, reg_covar=1e-6, cov_type="full", random_state=0)
    model = {
        "gmm": gmm,
        "tau": tau_med,
        "alpha_z": params["alpha_z"],
        "beta_z": params["beta_z"],
        "alpha_x": params["alpha_x"]
    }
    print(f"Basis Funcitons:{gmm.n_components} | Samples: {Z.shape[0]}")
    return model


def orientation_dmp_rollout_gmm(model, T=None, dt=0.01, q0=None, qg=None, tau=None):
    gmm = model["gmm"]
    alpha_z = model["alpha_z"]
    beta_z = model["beta_z"]
    alpha_x = model["alpha_x"]
    tau = float(model["tau"] if tau is None else tau)

    if T is None: T = tau
    if qg is None: qg = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if q0 is None: q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    e = quaternion_log(quaternion_multiply(qg.reshape(1,4), quaternion_conjugate(q0).reshape(1,4)))[0] * 2.0
    d = 2.0 * quaternion_log(quaternion_multiply(qg, quaternion_conjugate(q0)))
    D = np.diag(d)
    z = np.zeros(3, dtype=float)

    num_steps = int(round(T / dt))
    ts = np.linspace(0, T, num_steps + 1)
    qs = []

    for tcur in ts:
        x = np.exp(-alpha_x * tcur / tau)
        y_mean = gmr_predict(gmm, np.array([[x]]))

        f = (y_mean[0] * x)

        z_dot = (-alpha_z*(beta_z*e + z) + D @ f) / tau
        e_dot = z / tau
        z += z_dot * dt
        e += e_dot * dt

        delta = q_exp(0.5 * e.reshape(1,3))[0]
        #q = quaternion_multiply(delta.reshape(1,4), qg.reshape(1,4))[0]
        q = quaternion_multiply(quaternion_conjugate(delta.reshape(1,4)), qg.reshape(1,4))[0]

        q = q_normalize(q)
        qs.append(q)

    return ts, np.stack(qs, axis=0)


if __name__ == "__main__":

    input_glob = r"C:\Users\ceaz\OneDrive\Desktop\Code for thesis\recordings2\*NewOri*.json"
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No files match: {input_glob}")
    print(f"[load] Found {len(files)} files")

    demos = []
    joint_names = None
    for p in files:
        t, pos_raw, jnames = load_demo_json(p)
        t, pos_clean = clean_positions_with_constraints(t, pos_raw, jnames)
        # --- START: Trimming code ---
        print(f"Processing: {os.path.basename(p)}")

        # Select a joint to monitor for movement
        try:
            move_joint_idx = jnames.index("Right_Wrist")
        except ValueError:
            print("Warning: 'Right_Wrist' not found. Defaulting to joint 0 for trimming check.")
            move_joint_idx = 0

        joint_pos = pos_clean[:, move_joint_idx, :]

        if len(joint_pos) == 0:
            print(f"Skipping empty demo: {p}")
            continue

        dist_from_start = np.linalg.norm(joint_pos - joint_pos[0], axis=1)
        movement_threshold = 0.01
        start_indices = np.where(dist_from_start > movement_threshold)[0]

        start_idx = 0
        if len(start_indices) > 0:
            start_idx = start_indices[0]
            print(f"Movement detected at index {start_idx}. Trimming {start_idx} static frames.")
        else:
            print(f"Warning: No significant movement detected in file. Using full demo.")

        # Trim the data arrays
        t = t[start_idx:]
        pos_clean = pos_clean[start_idx:, :, :]

        # Re-normalize time to start at 0
        if len(t) > 0:
            t = t - t[0]
        else:
            print(f"Skipping demo with no data after trimming: {p}")
            continue

        if joint_names is None:
            joint_names = jnames
        elif jnames != joint_names:
            raise ValueError(f"Joint order mismatch in {p}")

        pos_s = smooth_positions(pos_clean)
        q_xyzw = calculate_hand_orientation(pos_s, jnames)   # returns (T,4) in xyzw
        q_wxyz = np.roll(q_xyzw, shift=1, axis=-1)           # to wxyz
        q_wxyz = q_normalize(q_wxyz)
        q_wxyz = enforce_hemisphere(q_wxyz)

        demos.append({
            "file": os.path.basename(p),
            "t": t,
            "pos": pos_clean,
            "joint_names": joint_names,
            "quat_wxyz": q_wxyz
        })
    print(f"[load] Loaded & cleaned {len(demos)} demos")

    j_idx, _ = pick_joint_index(joint_names, "Right_Wrist")
    ref_demo = pick_reference_demo(demos, j_idx)
    print(f"[ref] Using '{ref_demo['file']}' as DTW reference")


    model = fit_orientation_dmp_gmm(
        demos,
        use_dtw=True,
        ref_demo=ref_demo,
        alpha_z=20.0,
        beta_z=None,
        alpha_x=math.log(100.0)
    )
    print(f"[fit] GMM components: {model['gmm'].n_components} | tau(median): {model['tau']:.3f}s")
    meta = {
        "files": [d["file"] for d in demos],
        "ref_demo": ref_demo["file"],
        "joint_basis": ["Right_Wrist", "Right_Thumb_Tip", "Right_Index_Tip"]
    }
    save_orientation_model("orientation_dmp_gmm.json", model, meta=meta)

    t0, q_demo0 = quats_from_demo(demos[0])
    q0, qg = q_demo0[0], q_demo0[-1]
    T = float(model["tau"])
    dt = 0.01

    t_out, q_out_wxyz = orientation_dmp_rollout_gmm(
        model,
        T=T,
        dt=dt,
        q0=q0,
        qg=qg
    )

    t_ref, q_ref = quats_from_demo(ref_demo)
    # quick resample if lengths differ
    if len(t_ref) != len(t_out):
        idx = np.searchsorted(t_ref, np.clip(t_out, t_ref[0], t_ref[-1]))
        idx = np.clip(idx, 0, len(t_ref)-1)
        q_ref = q_ref[idx]
    plot_orientation_comparison(t_out, q_ref, q_out_wxyz, title=f"Orientation vs ref ({ref_demo['file']})")

    q_out_xyzw = np.roll(q_out_wxyz, shift=-1, axis=-1)
    plot_quaternion_components(t_out, q_out_xyzw, title="Orientation DMP rollout (components)", show=True, block=True)
    plot_orientation_components_split(
        t_out,
        q_ref,          # wxyz
        q_out_wxyz,     # wxyz
        title=f"Orientation vs ref (split) — {ref_demo['file']}",
        show=True,
        block=True
    )
    fig = plot_orientation_overview(t_out, q_ref, q_out_wxyz,  alpha_x=model["alpha_x"], title="Orientation overview")
    #fig_time  = overlay_orientation_demos(demos, mode="time",  alpha_x=model["alpha_x"])
    #fig_phase = overlay_orientation_demos(demos, mode="phase", alpha_x=model["alpha_x"])
    #fig_dtw   = overlay_orientation_demos(demos, mode="dtw",   alpha_x=model["alpha_x"])
    #plot_axis_trajectory_on_sphere(q_ref,  title="Ref axis (z) path")
    #plot_axis_trajectory_on_sphere(q_out_wxyz, title="DMP axis (z) path")
    #plot_axis_trajectories_on_sphere_overlay(q_ref_wxyz=q_ref, q_dmp_wxyz=q_out_wxyz, axis='z', title="Ref vs DMP on unit sphere")

    plt.show()

