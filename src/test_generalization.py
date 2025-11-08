import os, glob, math, json, time
import numpy as np
import matplotlib.pyplot as plt
from cleaning_data import load_demo_json, clean_positions_with_constraints
from helper import (
    smooth_positions, kinematics, forcing_target,
    canonical_phase, pick_joint_index, pick_reference_demo,
    dtw_warp_indices, fit_gmm_bic, gmr_predict,
    calculate_hand_orientation # Added from helper
)
from orientation_DMP import (
    q_normalize, quaternion_multiply, quaternion_conjugate,
    quaternion_log, q_exp, enforce_hemisphere, derivatives_quat,
)

def quats_from_demo(demo):
    t = demo["t"]
    pos = demo["pos"]
    joint_names = demo["joint_names"]
    pos_s = smooth_positions(pos)
    q_xyzw = calculate_hand_orientation(pos_s, joint_names)
    q = np.roll(q_xyzw, shift=1, axis=-1)
    q = q_normalize(q)
    q = enforce_hemisphere(q)
    return t, q

def build_training_pairs_pos_dtw(demos, j_idx, ref_demo, alpha_y=80.0, beta_y=None, alpha_x=math.log(100), eps=1e-8):
    if beta_y is None:
        beta_y = alpha_y / 4.0
    pos_ref = smooth_positions(ref_demo["pos"])[:, j_idx, :]
    Xs, Ys, taus = [], [], []
    for demo in demos:
        t, pos = demo["t"], smooth_positions(demo["pos"])
        V, A = kinematics(pos, t)
        goal = pos[-1]
        y0 = pos[0]
        x_demo, tau =  canonical_phase(t, alpha_x=alpha_x)
        taus.append(max(t[-1] - t[0], 1e-12))
        f = forcing_target(pos, V, A, goal, tau, alpha_y=alpha_y, beta_y=beta_y)

        p_demo = smooth_positions(demo["pos"])[:, j_idx, :]
        idx_map = dtw_warp_indices(p_demo, pos_ref)
        f_wrap = f[idx_map, j_idx, :]
        x_wrap = x_demo[idx_map]
        mask = (x_wrap >= 0.02) & (x_wrap <= 0.98)

        if not np.any(mask): continue
        xw = x_wrap[mask][:, None]
        fw = f_wrap[mask]
        k_pos = goal[j_idx, :] - y0[j_idx, :]


        k_pos_safe = k_pos + 1e-8
        fw_norm = fw / k_pos_safe
        yw = fw_norm / (xw + eps)

        good = np.isfinite(yw).all(axis=-1)
        if np.any(good):
            Xs.append(xw[good])
            Ys.append(yw[good])

    X = np.vstack(Xs)
    Y = np.vstack(Ys)
    tau_median = float(np.median(taus))
    params = dict(alpha_y=alpha_y, beta_y=beta_y, alpha_x=alpha_x, tau_median=tau_median)
    return X, Y, params, tau_median

def fit_position_dmp_gmm(demos, j_idx, demo, alpha_y=80.0, beta_y=None, alpha_x=math.log(100)):
    X, Y, params, tau_median = build_training_pairs_pos_dtw(demos, j_idx, demo,
                                                            alpha_y=alpha_y, beta_y=beta_y, alpha_x=alpha_x)
    Z = np.hstack([X, Y])
    finite = np.isfinite(Z).all(axis=1)
    Z = Z[finite]
    gmm = fit_gmm_bic(Z, Ks=range(2, 70), n_init=8, reg_covar=1e-4, cov_type="full", random_state=0)
    model = {
        "gmm": gmm, "tau": tau_median, "alpha_y": params["alpha_y"],
        "beta_y": params["beta_y"], "alpha_x": params["alpha_x"],
    }
    print(f"[GMM-pos] GMM components: {gmm.n_components} | samples={Z.shape[0]}")
    return model

def position_dmp_rollout_gmm(model, T=None, dt=0.01, y0=None, g=None, tau=None):
    gmm = model["gmm"]
    alpha_y = model["alpha_y"]
    beta_y  = model["beta_y"]
    alpha_x = model["alpha_x"]
    tau = float(model["tau"] if tau is None else tau)
    if T is None: T = tau
    k_pos = g - y0
    k_pos_safe = k_pos + 1e-8
    y = y0.astype(float).copy()
    z = np.zeros(3, dtype=float)
    ts = np.arange(0, T+dt, dt)
    ys = []
    for tcur in ts:
        x = np.exp(-alpha_x * tcur / tau)
        y_mean = gmr_predict(gmm, np.array([[x]]))

        f_norm = y_mean[0] * x
        f = f_norm * k_pos_safe
        z_dot = (alpha_y * (beta_y * (g - y) - z) + f) / tau

        y_dot = z / tau
        z += z_dot * dt
        y += y_dot * dt
        ys.append(y.copy())
    return ts, np.stack(ys, axis=0)

# --- Functions from LWR Script (Script 9) ---
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

        pos_demo_joint = pos_smoothed[:, j_idx, :]
        idx_map = dtw_warp_indices(pos_demo_joint, pos_ref_joint)
        x_warp = x_demo[idx_map]
        f_pos_warp = f_pos[idx_map, j_idx, :]

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

    X_pos, Y_pos = np.vstack(Xs_pos), np.vstack(Ys_pos)
    print(f"[LWR-pos] Model prepared with {len(X_pos)} data points.")
    tau_median = float(np.median(taus))
    pos_model = { "X": X_pos, "Y": Y_pos, "tau": tau_median, "alpha_x": alpha_x, "alpha_y": alpha_y, "beta_y": beta_y }
    return pos_model

def lwr_predict(lwr_model, x_query, kernel_width=0.04):
    X_train, Y_train = lwr_model['X'], lwr_model['Y']
    distances = np.abs(X_train - x_query)
    weights = np.exp(-(distances**2) / (2 * kernel_width**2))
    numerator = np.sum(weights * Y_train, axis=0)
    denominator = np.sum(weights, axis=0) + 1e-9
    return numerator / denominator

def position_dmp_rollout_lwr(model, T=None, dt=0.01, y0=None, g=None, tau=None, kernel_width_pos=0.04):
    alpha_x = model["alpha_x"]
    alpha_y, beta_y = model["alpha_y"], model["beta_y"]
    tau = float(model["tau"] if tau is None else tau)
    if T is None: T = tau
    y = np.copy(y0)
    z_pos = np.zeros_like(y)
    k_pos = g - y0
    k_pos_safe = k_pos + 1e-8
    ts = np.arange(0, T + dt, dt)
    position_trajectory = []
    for t in ts:
        x = np.exp(-alpha_x * t / tau)
        f_pos = np.zeros(3)
        if x > 0.01:
            y_shape_pos_norm = lwr_predict(model, x, kernel_width=kernel_width_pos)
            f_pos_norm = y_shape_pos_norm * x
        else:
            f_pos_norm = np.zeros(3)
        f_pos = f_pos_norm * k_pos_safe
        z_dot_pos = (alpha_y * (beta_y * (g - y) - z_pos) + f_pos) / tau
        z_pos += z_dot_pos * dt
        y += (z_pos / tau) * dt
        position_trajectory.append(np.copy(y))
    return ts, np.array(position_trajectory)


def rollout_plain_pos_dmp(y0, g, T, dt=0.01, alpha_y=25.0, beta_y=None, tau=None):
    if beta_y is None:
        beta_y = alpha_y / 4.0
    if tau is None:
        tau = T
    ts = np.arange(0, T + dt, dt)
    y = y0.copy().astype(float)
    z = np.zeros_like(y)
    out = []
    for t in ts:
        z_dot = (alpha_y * (beta_y * (g - y) - z)) / tau
        y_dot = z / tau
        z += z_dot * dt
        y += y_dot * dt
        out.append(y.copy())
    return ts, np.stack(out, axis=0)

def load_and_prepare_demos(glob_pattern, ee_name, trim_threshold=0.01):
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files match: {glob_pattern}")
    print(f"[load] Found {len(files)} files")
    demos = []
    joint_names = None
    for p in files:
        t, pos_raw, jnames = load_demo_json(p)
        t_clean, pos_clean = clean_positions_with_constraints(t, pos_raw, jnames)

        try:
            idx_ee = jnames.index(ee_name)
        except ValueError:
            print(f"Error: EE_JOINT_NAME '{ee_name}' not in joint list: {jnames}")
            return [], None

        pos_ee = smooth_positions(pos_clean)[:, idx_ee, :]
        dist = np.linalg.norm(pos_ee - pos_ee[0], axis=1)
        moving = np.where(dist > trim_threshold)[0]
        start_idx = 0
        if len(moving) > 0:
            start_idx = moving[0]

        t_final   = t_clean[start_idx:] - t_clean[start_idx]
        pos_final = pos_clean[start_idx:, :, :]

        if len(t_final) < 10:
            print(f"[warn] skipping {os.path.basename(p)}: too short")
            continue
        if joint_names is None:
            joint_names = jnames
        elif jnames != joint_names:
            raise ValueError(f"Joint order mismatch in {p}")
        demos.append({
            "file": os.path.basename(p), "t": t_final,
            "pos": pos_final, "joint_names": jnames,
        })
    print(f"[load] Loaded & prepped {len(demos)} demos")
    return demos, joint_names

#def plot_generalization_3d(ax, demos, j_idx, P_gmm_new, P_lwr_new, P_plain):
#    """Plots the 3D comparison with no legend, as requested."""
#
#    # 1. Plot all background demos
#    for d in demos:
#        pos = smooth_positions(d["pos"])[:, j_idx, :]
#        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color='#AAAAAA', alpha=0.35, linewidth=1.5)
#
#    # 2. Plot the GMM generated trajectory
#    ax.plot(P_gmm_new[:, 0], P_gmm_new[:, 1], P_gmm_new[:, 2],
#            color='blue', linewidth=2.5, linestyle='--')
#
#    # 3. Plot the LWR generated trajectory
#    ax.plot(P_lwr_new[:, 0], P_lwr_new[:, 1], P_lwr_new[:, 2],
#            color='green', linewidth=2.5, linestyle='--')
#
#    # 4. Plot the plain (no forcing term) trajectory
#    ax.plot(P_plain[:, 0], P_plain[:, 1], P_plain[:, 2],
#            color='red', linewidth=2.0, linestyle=':')
#
#    ax.set_xlabel("X")
#    ax.set_ylabel("Y")
#    ax.set_zlabel("Z")
    # No legend
def plot_generalization_3d(ax, demos, j_idx,
                           P_gmm_new, P_lwr_new, P_plain,
                           P_gmm_orig, P_lwr_orig,
                           y0_orig=None, g_orig=None, y0_new=None, g_new=None):

    # 1) background demos
    for d in demos:
        pos = smooth_positions(d["pos"])[:, j_idx, :]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                color='#AAAAAA', alpha=0.35, linewidth=1.5)

    # 2) GMM: original (solid) vs new (dashed)
    ax.plot(P_gmm_orig[:, 0], P_gmm_orig[:, 1], P_gmm_orig[:, 2],
            color='blue', linewidth=2.5, linestyle='-', label='GMM (orig)')
    ax.plot(P_gmm_new[:, 0],  P_gmm_new[:, 1],  P_gmm_new[:, 2],
            color='blue', linewidth=2.5, linestyle='--', label='GMM (new)')

    # 3) LWR: original (solid) vs new (dashed)
    ax.plot(P_lwr_orig[:, 0], P_lwr_orig[:, 1], P_lwr_orig[:, 2],
            color='green', linewidth=2.5, linestyle='-', label='LWR (orig)')
    ax.plot(P_lwr_new[:, 0],  P_lwr_new[:, 1],  P_lwr_new[:, 2],
            color='green', linewidth=2.5, linestyle='--', label='LWR (new)')

    # 4) Plain spring-damper to the new goal
    ax.plot(P_plain[:, 0], P_plain[:, 1], P_plain[:, 2],
            color='red', linewidth=2.0, linestyle=':', label='Plain (new)')

    # 5) start/goal markers
    if y0_orig is not None:
        ax.scatter(*y0_orig, c='blue',  s=50, marker='o', label='Start (orig)')
    if g_orig is not None:
        ax.scatter(*g_orig,  c='blue',  s=120, marker='x', label='Goal (orig)')
    if y0_new is not None:
        ax.scatter(*y0_new, c='green', s=50, marker='o', label='Start (new)')
    if g_new is not None:
        ax.scatter(*g_new,  c='black', s=150, marker='x', label='Goal (new)')

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass

    #handles, labels = ax.get_legend_handles_labels()
    #uniq = dict(zip(labels, handles))
    #ax.legend(uniq.values(), uniq.keys(), loc='best')

if __name__ == "__main__":

    # --- Hyperparameters ---
    INPUT_GLOB = r"/home/ceazar/src/skilllearninglib/data/recordings/movingNew3.json"
    EE_JOINT_NAME = "Right_Wrist"
    PLOT = True

    DELTA_START = np.array([0.0, 0.4, 0.0])
    DELTA_GOAL  = np.array([0.2, 0.3, 0.0])

    demos, joint_names = load_and_prepare_demos(INPUT_GLOB, EE_JOINT_NAME)
    if not demos:
        raise RuntimeError("No demos to train on.")

    j_idx, j_name = pick_joint_index(joint_names, EE_JOINT_NAME)
    ref_demo = pick_reference_demo(demos, j_idx)
    print(f"[main] EE joint: {j_name} (idx={j_idx})")
    print(f"[main] Reference demo: {ref_demo['file']}")

    P_ref_smt = smooth_positions(ref_demo["pos"])
    y0_orig = P_ref_smt[0, j_idx, :]
    g_orig  = P_ref_smt[-1, j_idx, :]

    y0_new = y0_orig + DELTA_START
    g_new  = g_orig  + DELTA_GOAL
    single_demo = None
    for d in demos:
        if d["file"] == "movingNew3.json":
            single_demo = d
            break
    if single_demo is None:
        print("[warn] movingNew3.json not found among loaded demos; skipping single-demo LWR.")
    else:
        P_single_smt = smooth_positions(single_demo["pos"])
        y0_single_orig = P_single_smt[0, j_idx, :]
        g_single_orig  = P_single_smt[-1, j_idx, :]
        y0_single_new  = y0_single_orig + DELTA_START
        g_single_new   = g_single_orig  + DELTA_GOAL

    print(f"Original Goal: {g_orig}")
    print(f"New Goal:      {g_new}")

    print("\n--- Training GMM Model ---")
    pos_model_gmm = fit_position_dmp_gmm(
        demos, j_idx, ref_demo, alpha_y=25.0
    )
    T = pos_model_gmm["tau"]

    print("\n--- Training LWR Model ---")
    pos_model_lwr = prepare_lwr_models(
        demos, ref_demo, j_idx, alpha_y=25.0
    )

    if single_demo is not None:
        pos_model_lwr_single = prepare_lwr_models(
            [single_demo], single_demo, j_idx, alpha_y=25.0
        )

    print("\n--- Running Rollouts ---")

    t, P_plain = rollout_plain_pos_dmp(
        y0_new, g_new, T=T, dt=0.01,
        alpha_y=pos_model_gmm["alpha_y"],
        beta_y=pos_model_gmm["beta_y"],
        tau=T
    )

    _, P_gmm_orig = position_dmp_rollout_gmm(
        pos_model_gmm, T=T, dt=0.01, y0=y0_orig, g=g_orig, tau=T
    )

    _, P_gmm_new = position_dmp_rollout_gmm(
        pos_model_gmm, T=T, dt=0.01, y0=y0_new, g=g_new, tau=T
    )

    _, P_lwr_orig = position_dmp_rollout_lwr(
        pos_model_lwr, T=T, dt=0.01, y0=y0_orig, g=g_orig, tau=T
    )

    _, P_lwr_new = position_dmp_rollout_lwr(
        pos_model_lwr, T=T, dt=0.01, y0=y0_new, g=g_new, tau=T
    )
    if single_demo is not None:
        T_single = pos_model_lwr_single["tau"]

    _, P_lwr1_orig = position_dmp_rollout_lwr(
        pos_model_lwr_single, T=T_single, dt=0.01,
        y0=y0_single_orig, g=g_single_orig, tau=T_single
    )

    _, P_lwr1_new = position_dmp_rollout_lwr(
        pos_model_lwr_single, T=T_single, dt=0.01,
        y0=y0_single_new, g=g_single_new, tau=T_single
    )
    print("All rollouts complete.")

    print("\n--- Model Comparison Metrics ---")

    gmm_final_err_orig = np.linalg.norm(P_gmm_orig[-1] - g_orig)
    gmm_final_err_new  = np.linalg.norm(P_gmm_new[-1] - g_new)

    lwr_final_err_orig = np.linalg.norm(P_lwr_orig[-1] - g_orig)
    lwr_final_err_new  = np.linalg.norm(P_lwr_new[-1] - g_new)

    plain_final_err_new = np.linalg.norm(P_plain[-1] - g_new)


    gmm_shape_err = np.mean(np.linalg.norm(P_gmm_new - P_plain, axis=1))
    lwr_shape_err = np.mean(np.linalg.norm(P_lwr_new - P_plain, axis=1))

    print("      GENERALIZATION TEST RESULTS (Position)")
    print(f"Goal changed by: {DELTA_GOAL} (norm = {np.linalg.norm(DELTA_GOAL):.3f} m)")

    print("---------------------------------------------------------")
    print("| Model | Original Goal Error (m) | New Goal Error (m) |")
    print("---------------------------------------------------------")
    print(f"| GMM   | {gmm_final_err_orig:<23.5f} | {gmm_final_err_new:<18.5f} |")
    print(f"| LWR   | {lwr_final_err_orig:<23.5f} | {lwr_final_err_new:<18.5f} |")
    print(f"| Plain | N/A                       | {plain_final_err_new:<18.5f} |")
    print("---------------------------------------------------------")


    print("---------------------------------------------------------")
    print("| Model | Avg. Shape Difference (m) |")
    print("---------------------------------------------------------")
    print(f"| GMM  | {gmm_shape_err:<25.5f} |")
    print(f"| LWR   | {lwr_shape_err:<25.5f} |")
    print("---------------------------------------------------------")

    if PLOT:
        fig = plt.figure(figsize=(12, 8))
        ax  = fig.add_subplot(1, 1, 1, projection='3d')

        plot_generalization_3d(
            ax=ax,
            demos=demos,
            j_idx=j_idx,
            P_gmm_new=P_gmm_new,
            P_lwr_new=P_lwr_new,
            P_plain=P_plain,
            P_gmm_orig=P_gmm_orig,
            P_lwr_orig=P_lwr_orig,
            y0_orig=y0_orig, g_orig=g_orig,
            y0_new=y0_new,   g_new=g_new
        )
    if single_demo is not None:
        ax.plot(
            P_lwr1_orig[:, 0], P_lwr1_orig[:, 1], P_lwr1_orig[:, 2],
            color='orange', linewidth=2.5, linestyle='-', label='Linear Regression (orig)'
        )
        ax.plot(
            P_lwr1_new[:, 0], P_lwr1_new[:, 1], P_lwr1_new[:, 2],
            color='orange', linewidth=2.5, linestyle='--', label='Linear Regression (new)'
        )
        ax.scatter(*y0_single_orig, c='orange', s=40, marker='o', label='Start (LR-1)')
        ax.scatter(*g_single_orig,  c='orange', s=90, marker='x', label='Goal (LR-1)')
    ax.scatter(g_orig[0], g_orig[1], g_orig[2], c='gray', s=100, marker='x', label="Original Goal")
    ax.scatter(g_new[0],  g_new[1],  g_new[2],  c='black', s=150, marker='x', label="New Goal")
    ax.set_title("Generalization to New Goal")
    #ax.legend()
    handles, labels = ax.get_legend_handles_labels()

    uniq = dict(zip(labels, handles))

    ax.legend(uniq.values(), uniq.keys(),
    loc='center left',
    bbox_to_anchor=(-0.1, 0.5))
    plt.show()