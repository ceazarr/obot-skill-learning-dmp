import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import savgol_filter

def load_data(path):
    with open(path, 'r') as file:
        data = json.load(file)
    # Extract data
    joint_names = [k for k in data[0] if k not in ['frame', 'timestamp']]
    num_joints = len(joint_names)
    num_frames = len(data)

    # Build position array
    positions = np.zeros((num_frames, num_joints, 3), dtype=np.float32)
    for i, frame in enumerate(data):
        for j, joint in enumerate(joint_names):
            positions[i][j] = frame[joint]

    timestamps = np.array([frame['timestamp'] for frame in data])

    return timestamps, joint_names, positions

def smooth_positions(pos, window_length=11, polyorder=5, sigma=1.5):
    T = pos.shape[0]
    window_length = min(window_length, T if T % 2 == 1 else T - 1)
    if window_length < 5:
        window_length = 5

    out = np.empty_like(pos)
    J = pos.shape[1]

    for j in range(J):
        for coord in range(3):
            try:
                out[:, j, coord] = savgol_filter(
                    pos[:, j, coord],
                    window_length=window_length,
                    polyorder=polyorder,
                    mode='nearest'
                )
            except:
                # Fallback to Gaussian smoothing if Savgol fails
                out[:, j, coord] = ndimage.gaussian_filter1d(
                    pos[:, j, coord],
                    sigma=sigma
                )
    return out

def compute_kinematics(pos, t, eps=1e-6):
    t = np.asarray(t, float)
    t = np.maximum.accumulate(t)
    dt = np.diff(t)
    dt[dt < eps] = eps
    vel = np.zeros_like(pos); acc = np.zeros_like(pos)
    
    vel[1:] = np.diff(pos, axis=0) / dt[:, None, None]
    vel[0] = vel[1]
    acc[1:] = np.diff(vel, axis=0) / dt[:, None, None]
    acc[0] = acc[1]

    return vel, acc, dt

def compute_phase(t, alpha_x=4.605170185988092): # alpha_x is like that as this nimber makes x(1) = 0.01 (1%)
    T_total = max(t[-1] - t[0], 1e-12)
    t_rel = (t - t[0]) / T_total
    x = np.exp(-alpha_x * t_rel)
    return x, t_rel, T_total

def build_basis_x(x, N=30, overlap_factor=0.5):
    T = len(x)
    idx = np.linspace(0, T - 1, N, dtype=int)
    c = x[idx]
    dc = np.diff(c)
    avg_c = np.mean(np.abs(dc)) if len(dc) > 0 else 1.0
    avg_c = max(avg_c, 1e-8)
    overlap_factor = 0.5 # for smoother interpolation
    h = np.full(N, 1.0 / (overlap_factor * avg_c) ** 2)
    diff = x[:, None] - c[None, :]         # (x - c)
    psi = np.exp(-h[None, :] * diff ** 2)  # exp(-h * (x - c) ** 2)
    den = psi.sum(axis=1, keepdims=True) + 1e-8
    psi /= den

    return psi, c, h

def compute_forcing_target(pos, vel, acc, g, alpha_y=25, beta_y=None):
    if beta_y is None:
        beta_y = alpha_y / 4.0
    f = acc - alpha_y * (beta_y * (g[None, :, :] - pos) - vel)

    return f

def solve_ridge(Phi, target_fx, lambda_reg=1e-5):
    T, J, _ = target_fx.shape
    Y = target_fx.reshape(T, J * 3)
    A = Phi.T @ Phi + lambda_reg * np.eye(Phi.shape[1])
    B = Phi.T @ Y
    W = np.linalg.solve(A, B)

    return W

# def eval_metrics(f_true: np.ndarray, f_pred: np.ndarray):
#     if f_true.shape != f_pred.shape:
#         raise ValueError("Shape mismatch for metrics.")
#     mse = np.mean((f_true - f_pred) ** 2)
#     per_joint = np.mean((f_true - f_pred) ** 2, axis=0)  # (J,3)
#     denom = np.mean(f_true ** 2) + 1e-12
#     rel = mse / denom
# 
#     return float(mse), per_joint, float(rel)

def compute_total_displacement(positions_s: np.ndarray):
    diffs = np.diff(positions_s, axis=0)
    seg_len = np.linalg.norm(diffs, axis=2)  # (T-1, J)

    return seg_len.sum(axis=0)

def save_params(path, W, alpha_y, beta_y, alpha_x, g, start, duration, centers, widths, joint_names, mse):
    payload = {
        "weights": W.tolist(),
        "alpha_y": float(alpha_y),
        "beta_y": float(beta_y),
        "alpha_x": float(alpha_x),
        "goal": g.tolist(),
        "start": start.tolist(),
        "duration": float(duration),
        "basis_centers_x": centers.tolist(),
        "basis_widths": widths.tolist(),
        "joint_names": joint_names,
        "reconstruction_error": float(mse),
    }
    with open(path, 'w') as file:
        json.dump(payload, file, indent=4)

def generate_dmp_trajectory(W, centers, widths, start_pos, goal_pos, duration, 
                           alpha_y=25.0, beta_y=None, alpha_x=4.605170185988092, 
                           dt=0.01):
    if beta_y is None:
        beta_y = alpha_y / 4.0
    
    # Time steps
    t = np.arange(0, duration + dt, dt)
    T = len(t)
    
    # Phase variable
    x = np.exp(-alpha_x * t / duration)
    
    # Basis functions
    diff = x[:, None] - centers[None, :]
    psi = np.exp(-widths[None, :] * diff ** 2)
    psi /= (psi.sum(axis=1, keepdims=True) + 1e-8)
    
    # Initialize trajectory arrays
    J = start_pos.shape[0]  # number of joints
    pos = np.zeros((T, J, 3))
    vel = np.zeros((T, J, 3))
    acc = np.zeros((T, J, 3))
    
    # Initial conditions
    pos[0] = start_pos
    vel[0] = 0
    
    # Integration
    for i in range(1, T):
        # Forcing function
        f = (psi[i-1:i] @ W).reshape(J, 3) * x[i-1]
        
        # DMP equations
        acc[i] = alpha_y * (beta_y * (goal_pos - pos[i-1]) - vel[i-1]) + f
        vel[i] = vel[i-1] + acc[i] * dt
        pos[i] = pos[i-1] + vel[i] * dt
    
    return t, pos, vel, acc

def _basis_from_x(x, centers, widths):
    diff = x[:, None] - centers[None, :]
    psi = np.exp(-widths[None, :] * diff**2)
    psi /= (psi.sum(axis=1, keepdims=True) + 1e-8)
    return psi

def generate_dmp_trajectory_on_t(W, centers, widths, start_pos, goal_pos, t,
                                 alpha_y=25.0, beta_y=None, alpha_x=4.605170185988092):
    if beta_y is None:
        beta_y = alpha_y / 4.0
    t = np.asarray(t, float)
    T = len(t)
    x = np.exp(-alpha_x * (t - t[0]) / max(t[-1] - t[0], 1e-12))
    psi = _basis_from_x(x, centers, widths)

    J = start_pos.shape[0]
    pos = np.zeros((T, J, 3)); vel = np.zeros_like(pos); acc = np.zeros_like(pos)
    pos[0] = start_pos; vel[0] = 0

    dt = np.diff(t, prepend=t[0])
    if T > 1 and dt[0] == 0:
        dt[0] = dt[1]

    for i in range(1, T):
        f = (psi[i-1:i] @ W).reshape(J, 3) * x[i-1]
        acc[i] = alpha_y * (beta_y * (goal_pos - pos[i-1]) - vel[i-1]) + f
        vel[i] = vel[i-1] + acc[i] * dt[i]
        pos[i] = pos[i-1] + vel[i] * dt[i]
    return pos

def position_metrics(ts, pos_s, W, centers, widths, g,
                     alpha_y=25.0, beta_y=None, alpha_x=4.605170185988092):
    pos_gen = generate_dmp_trajectory_on_t(W, centers, widths, pos_s[0], g, ts,
                                           alpha_y=alpha_y, beta_y=beta_y, alpha_x=alpha_x)
    err = pos_s - pos_gen
    mse_global = float(np.mean(err**2))              # GLOBAL MSE over time × joints × XYZ
    mse_time   = np.mean(err**2, axis=(1, 2))        # per-time MSE
    mse_per_j  = np.mean(err**2, axis=0)             # (J, 3)
    return mse_global, mse_time, mse_per_j, pos_gen


def train_dmp_weights(json_path: str,
                      window_length: int = 11,
                      polyorder: int = 5,
                      sigma: float = 1.5,
                      alpha_y: float = 25.0,
                      beta_y: float = None,
                      alpha_x: float = 4.605170185988092,  # ln(100)
                      N: int = 30,
                      overlap_factor: float = 0.5,
                      mode: str = "canonical"):
    # 1) Load
    ts, joint_names, pos = load_data(json_path)

    # 2) Smooth
    pos_s = smooth_positions(pos, window_length=window_length, polyorder=polyorder, sigma=sigma)

    # 3) Kinematics
    vel, acc, dt = compute_kinematics(pos_s, ts)

    # 4) Gains & goal
    if beta_y is None:
        beta_y = alpha_y / 4.0
    g = pos_s[-1].copy() # Goal is the last position

    # 5) Forcing target
    f_target = compute_forcing_target(pos_s, vel, acc, g, alpha_y, beta_y)

    # 6) Phase
    x, t_rel, T_total= compute_phase(ts, alpha_x=alpha_x)
    mask = (x > 0.01)  & (x < 0.99)  # avoid edges where x is too close to 0 or 1
    
    # 7) Basis & training target
    Phi, centers, widths = build_basis_x(x, N=N)
    Phi_m = Phi[mask]
    f_train = (f_target / (x[:, None, None] + 1e-8))[mask]

    # 8) Ridge solve
    W = solve_ridge(Phi_m, f_train, lambda_reg=1e-5)

    # 9) Reconstruct forcing
    f_fit = (Phi @ W).reshape(len(ts), len(joint_names), 3)
    f_recon = f_fit * x[:, None, None]

    # 10) Metrics
    mse, mse_t, per_joint_mse, pos_gen = position_metrics(
        ts, pos_s, W, centers, widths, g,
        alpha_y=alpha_y, beta_y=beta_y, alpha_x=alpha_x
    )


    return {
        "weights": W, "Phi": Phi, "centers": centers, "widths": widths,
        "timestamps": ts, "joint_names": joint_names, "positions_s": pos_s,
        "velocity": vel, "acceleration": acc, "goal": g,
        "alpha_y": alpha_y, "beta_y": beta_y, "alpha_x": alpha_x, "x": x,
        "T_total": T_total, "f_target": f_target, "f_recon": f_recon,
        "mse": mse, "per_joint_mse": per_joint_mse, "mse_time": mse_t,
    }


def plot_results(res):
    ts   = res["timestamps"]
    Phi  = res["Phi"]
    x    = res["x"]
    f_t  = res["f_target"]
    f_r  = res["f_recon"]
    pos  = res["positions_s"]
    names = res["joint_names"]

    # pick a joint to visualize: Right_Wrist if present, else most active
    try:
        j = names.index("Right_Wrist")
    except ValueError:
        disp = (np.linalg.norm(np.diff(pos, axis=0), axis=2)).sum(axis=0)
        j = int(np.argmax(disp))

    # Generate DMP trajectory for comparison
    t_gen, pos_gen, vel_gen, acc_gen = generate_dmp_trajectory(
        res["weights"], res["centers"], res["widths"],
        res["positions_s"][0], res["goal"], res["T_total"],
        res["alpha_y"], res["beta_y"], res["alpha_x"]
    )

    fig = plt.figure(figsize=(18, 10))
    
    # Original 5 plots
    axes = [fig.add_subplot(2,4,k+1) for k in range(5)]

    # 1) Phase x(t)
    axes[0].plot(ts, x, linewidth=2)
    axes[0].set_title("Phase x(t)")
    axes[0].set_xlabel("time (s)"); axes[0].set_ylabel("x")

    # 2) Basis activations
    im = axes[1].imshow(Phi.T, aspect="auto", origin="lower")
    axes[1].set_title("Basis activations (Φ)")
    axes[1].set_xlabel("time index"); axes[1].set_ylabel("basis #")
    fig.colorbar(im, ax=axes[1])

    # 3) Forcing target vs reconstruction for one joint
    for d, lbl in enumerate(["X","Y","Z"]):
        axes[2].plot(ts, f_t[:, j, d], alpha=0.85, label=f"target {lbl}")
        axes[2].plot(ts, f_r[:, j, d], "--", label=f"recon {lbl}")
    axes[2].set_title(f"Forcing: {names[j]}")
    axes[2].set_xlabel("time (s)"); axes[2].set_ylabel("force")
    axes[2].legend(loc="upper right")

    # 4) Error over time
    pos_gen_on_ts = generate_dmp_trajectory_on_t(
        res["weights"], res["centers"], res["widths"],
        res["positions_s"][0], res["goal"], ts,
        res["alpha_y"], res["beta_y"], res["alpha_x"]
    )
    err_t = np.mean((res["positions_s"] - pos_gen_on_ts)**2, axis=(1,2))
    axes[3].plot(ts, err_t, linewidth=2)
    axes[3].set_title("Position MSE over time")
    axes[3].set_xlabel("time (s)"); axes[3].set_ylabel("MSE")

    # 5) Position comparison for selected joint
    p_orig = pos[:, j, :]
    for d, lbl in enumerate(["X","Y","Z"]):
        axes[4].plot(ts, p_orig[:, d], alpha=0.85, label=f"orig {lbl}")
        axes[4].plot(t_gen, pos_gen[:, j, d], "--", label=f"DMP {lbl}")
    axes[4].set_title(f"Position: {names[j]}")
    axes[4].set_xlabel("time (s)"); axes[4].set_ylabel("position")
    axes[4].legend(loc="upper right")

    # 6) 3D trajectory comparison - MAIN VISUALIZATION
    ax3d = fig.add_subplot(2,4,6, projection="3d")
    
    # Original trajectory
    p_orig = pos[:, j, :]
    ax3d.plot(p_orig[:,0], p_orig[:,1], p_orig[:,2], 
              linewidth=3, color='blue', label="Original", alpha=0.8)
    
    # Generated DMP trajectory
    p_gen = pos_gen[:, j, :]
    ax3d.plot(p_gen[:,0], p_gen[:,1], p_gen[:,2], 
              linewidth=2, color='red', linestyle='--', label="DMP Generated", alpha=0.8)
    
    # Start and end points
    ax3d.scatter(p_orig[0,0], p_orig[0,1], p_orig[0,2], 
                s=100, color='green', label="Start", marker='o')
    ax3d.scatter(p_orig[-1,0], p_orig[-1,1], p_orig[-1,2], 
                s=100, color='purple', label="Goal", marker='s')
    
    ax3d.set_title(f"3D Trajectory Comparison: {names[j]}")
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.legend()

    # 7) Trajectory error over time
    ax_err = fig.add_subplot(2,4,7)
    
    # Interpolate generated trajectory to match original timestamps for comparison
    from scipy.interpolate import interp1d
    if len(t_gen) > 1 and len(ts) > 1:
        interp_funcs = [interp1d(t_gen, pos_gen[:, j, d], kind='linear', 
                                bounds_error=False, fill_value='extrapolate') 
                       for d in range(3)]
        pos_gen_interp = np.array([f(ts) for f in interp_funcs]).T
        
        traj_error = np.linalg.norm(p_orig - pos_gen_interp, axis=1)
        ax_err.plot(ts, traj_error, linewidth=2, color='orange')
        ax_err.set_title(f"3D Position Error: {names[j]}")
        ax_err.set_xlabel("time (s)")
        ax_err.set_ylabel("L2 error")
        ax_err.grid(True, alpha=0.3)
        
        # Add error statistics as text
        mean_err = np.mean(traj_error)
        max_err = np.max(traj_error)
        ax_err.text(0.02, 0.98, f"Mean: {mean_err:.4f}\nMax: {max_err:.4f}", 
                   transform=ax_err.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 8) Error distribution
    ax_hist = fig.add_subplot(2,4,8)
    if len(t_gen) > 1 and len(ts) > 1:
        ax_hist.hist(traj_error, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax_hist.set_title("Error Distribution")
        ax_hist.set_xlabel("L2 error")
        ax_hist.set_ylabel("Frequency")
        ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print comparison statistics
    print(f"\n=== Trajectory Comparison for {names[j]} ===")
    if len(t_gen) > 1 and len(ts) > 1:
        print(f"Mean trajectory error: {mean_err:.6f}")
        print(f"Max trajectory error: {max_err:.6f}")
        print(f"RMS trajectory error: {np.sqrt(np.mean(traj_error**2)):.6f}")
    print(f"Original trajectory length: {len(ts)} points")
    print(f"Generated trajectory length: {len(t_gen)} points")

if __name__ == "__main__":
    path = "C:/Users/ceaz/OneDrive/Desktop/Code for thesis/recordings/test1.json"

    res = train_dmp_weights(path, N=50, alpha_y=15.0, alpha_x=4.605170185988092, window_length=21)
    print("Weights:", res["weights"].shape, "(N x 3J)")
    print("Reconstruction MSE:", res["mse"],
      "Relative:", res["mse"] / (np.mean(res["positions_s"]**2) + 1e-12))
    plot_results(res)

    save_params(
        path="table_cleaning_dmp_params_c.json", ############## =======> don't forget to change the path as it saves for each demo
        W=res["weights"],
        alpha_y=res["alpha_y"],
        beta_y=res["beta_y"],
        alpha_x=res["alpha_x"],
        g=res["goal"],
        start=res["positions_s"][0],
        duration=res["T_total"],
        centers=res["centers"],
        widths=res["widths"],
        joint_names=res["joint_names"],
        mse=res["mse"]
    )