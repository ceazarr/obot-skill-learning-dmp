import  math
import numpy as np
from scipy.signal import savgol_filter
from scipy import ndimage
from scipy.spatial.distance import euclidean
from sklearn.mixture import GaussianMixture
from fastdtw import fastdtw
from gmr import GMM as GMR
from scipy.spatial.transform import Rotation as R
from orientation_DMP import q_exp, q_normalize, quaternion_log, enforce_hemisphere

def smooth_positions(pos, window_length=11, polyorder=3, sigma=1.5):
    T, J, _ = pos.shape
    wl = min(window_length, T if T % 2 == 1 else T-1)
    wl = max(wl, 5)
    out = np.empty_like(pos)
    for j in range(J):
        for d in range(3):
            try:
                out[:, j, d] = savgol_filter(pos[:, j, d], window_length=wl, polyorder=polyorder, mode="nearest")
            except Exception:
                out[:, j, d] = ndimage.gaussian_filter1d(pos[:, j, d], sigma=sigma, mode="nearest")
    return out

#def kinematics(pos, t):
#    vel = np.zeros_like(pos)
#    acc = np.zeros_like(pos)
#    dt = np.diff(t, prepend=t[0])
#    dt[0] = dt[1] if len(dt) > 1 else 1.0
#    vel[1:] = (pos[1:] - pos[:-1]) / dt[1:, None, None]
#    vel[0]  = vel[1]
#    acc[1:] = (vel[1:] - vel[:-1]) / dt[1:, None, None]
#    acc[0]  = acc[1]
#    return vel, acc

def kinematics(pos, t):
    dt = np.mean(np.diff(t)) if len(t) > 1 else 1.0

    T, J, _ = pos.shape
    window_length = 11
    polyorder = 3

    vel = np.zeros_like(pos)
    acc = np.zeros_like(pos)

    for j in range(J):
        for d in range(3):
            try:
                vel[:, j, d] = savgol_filter(pos[:, j, d], window_length=window_length, polyorder=polyorder, deriv=1, delta=dt, mode="nearest")
                acc[:, j, d] = savgol_filter(pos[:, j, d], window_length=window_length, polyorder=polyorder, deriv=2, delta=dt, mode="nearest")
            except Exception:
                vel[:, j, d] = np.gradient(pos[:, j, d], dt)
                acc[:, j, d] = np.gradient(vel[:, j, d], dt)

    return vel, acc

def forcing_target(pos, vel, acc, g, tau, alpha_y=25.0, beta_y=None):
    if beta_y is None: beta_y = alpha_y / 4.0
    f = (tau**2 * acc) - alpha_y * (beta_y * (g[None,:,:] - pos) - (tau *vel))
    return f

def build_training_pairs_for_joint(demos, j, alpha_y=25.0, beta_y=None, alpha_x=math.log(100.0), eps=1e-8):

    Xs = []
    Ys = []
    for d in demos:
        t, pos = d["t"], d["pos"]
        pos_s = smooth_positions(pos)
        vel, acc = kinematics(pos_s, t)
        g = pos_s[-1]
        x, tau = canonical_phase(t, alpha_x=alpha_x)
        f = forcing_target(pos_s, vel, acc, g, tau, alpha_y=alpha_y, beta_y=beta_y)  # (T,J,3)

        y = f[:, j, :] / (x[:, None] + eps)

        #x_min, x_max = 0.01, 0.99
        #mask = (x >= x_min) & (x <= x_max)

        Xs.append(x[:, None])
        Ys.append(y)

    X = np.vstack(Xs)  # (sum T, 1)
    Y = np.vstack(Ys)  # (sum T, 3)
    return X, Y

def canonical_phase(t, alpha_x=math.log(100.0)):
    Ttot = max(t[-1] - t[0], 1e-12)
    tau  = (t - t[0]) / Ttot
    x = np.exp(-alpha_x * tau)
    return x, Ttot


def pick_joint_index(joint_names, focus_joint="Right_Wrist"):
    if focus_joint in joint_names:
        return joint_names.index(focus_joint), focus_joint
    return 0, joint_names[0]

def total_displacement_for_joint(pos_s, j):
    diffs = np.diff(pos_s[:, j, :], axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))

def pick_reference_demo(demos, j):
    disps = []
    for d in demos:
        if d["pos"] is None or len(d["pos"]) == 0:
            print("Invalid demo data:", d)
            continue
        pos_s = smooth_positions(d["pos"])
        disps.append(total_displacement_for_joint(pos_s, j))
    order = np.argsort(disps)
    return demos[order[len(order)//2]]

def dtw_warp_indices(src_seq: np.ndarray, ref_seq: np.ndarray) -> np.ndarray:
    #s = np.std(ref_seq, axis=0) + 1e-9
    dist, path = fastdtw(src_seq , ref_seq , dist=euclidean)
    ref_len = len(ref_seq)

    buckets = [[] for _ in range(ref_len)]
    for i_src, i_ref in path:
        if 0 <= i_ref < ref_len:
            buckets[i_ref].append(i_src)

    idx_map = np.empty(ref_len, dtype=int)
    last = 0
    for i, cols in enumerate(buckets):
        if cols:
            last = int(np.median(cols))
        idx_map[i] = last
    idx_map = np.clip(idx_map, 0, len(src_seq) - 1)
    return idx_map

def build_training_pairs_for_joint_dtw(demos, j, ref_demo, alpha_y=25.0, beta_y=None, alpha_x=math.log(100.0), eps=1e-8):
    Xs, Ys = [], []

    p_ref = smooth_positions(ref_demo["pos"])
    p_ref = p_ref[:, j, :]  # (T_ref, 3)

    for d in demos:
        t, pos = d["t"], d["pos"]
        pos_s = smooth_positions(pos)
        vel, acc = kinematics(pos_s, t)
        g = pos_s[-1]


        x_demo, tau = canonical_phase(t, alpha_x=alpha_x)
        f = forcing_target(pos_s, vel, acc, g, tau, alpha_y=alpha_y, beta_y=beta_y)  # (T, J, 3)

        p_demo = pos_s[:, j, :]  # (T, 3)

        idx_map = dtw_warp_indices(p_demo, p_ref)  # (T_ref,)

        f_wrap = f[idx_map, j, :]  # (T_ref, 3)
        x_wrap = x_demo[idx_map]



        y = f_wrap / (x_wrap[:, None] + eps)
        Xs.append(x_wrap[:, None])
        Ys.append(y)

    X = np.vstack(Xs)
    Y = np.vstack(Ys)
    print("pos_s shape:", pos_s.shape)
    return X, Y


def fit_gmm_bic(Zs, Ks=range(2, 50), n_init=8, reg_covar=1e-4, cov_type="full", random_state=0):

    best, best_bic = None, np.inf
    for K in Ks:
        gmm = GaussianMixture(n_components=K, covariance_type=cov_type,
                              n_init=n_init, reg_covar=reg_covar, random_state=random_state)
        gmm.fit(Zs)
        bic = gmm.bic(Zs)
        if bic < best_bic:
            best, best_bic = gmm, bic
    return best


def gmr_predict(gmm_sklearn, x_query: np.ndarray):
    gmr_gmm = GMR(
        n_components=gmm_sklearn.n_components,
        priors=gmm_sklearn.weights_,
        means=gmm_sklearn.means_,
        covariances=gmm_sklearn.covariances_,
    )
    y_mean = gmr_gmm.predict([0], x_query)
    return y_mean

#def calculate_hand_orientation(pos_clean, joint_names):
#    wrist_idx = joint_names.index("Right_Wrist")
#    thumb_idx = joint_names.index("Right_Thumb_Tip")
#    index_idx = joint_names.index("Right_Index_Tip")
#
#    p_wrist = pos_clean[:, wrist_idx, :]
#    p_thumb = pos_clean[:, thumb_idx, :]
#    p_index = pos_clean[:, index_idx, :]
#
#    T = pos_clean.shape[0]
#    quaternions = np.zeros((T, 4))
#
#    for i in range(T):
#        #=======z axis: wrist to to mid point between Index and Thumb========#
#        center = 0.5 * (p_thumb[i] + p_index[i])
#        z_axis = center - p_wrist[i]
#        if np.linalg.norm(z_axis) < 1e-5:
#            if i > 0:
#                quaternions[i] = quaternions[i-1]
#            else:
#                quaternions[i] = np.array([0, 0, 0, 1])
#            continue
#        z_axis /= np.linalg.norm(z_axis)
#
#        #=======x axis: pointing outwards========#
#        y_hat_axis = p_index[i] - p_thumb[i]
#        y_axis = y_hat_axis - np.dot(y_hat_axis, z_axis) * z_axis
#        #y_axis = np.cross(y_hat_axis, z_axis)
#        if np.linalg.norm(y_axis) < 1e-5:
#            if i > 0:
#                quaternions[i] = quaternions[i-1]
#            else:
#                quaternions[i] = np.array([0, 0, 0, 1])
#            continue
#        y_axis /= np.linalg.norm(y_axis)
#
#        #=======y axis: cross product========#
#        x_axis = np.cross(y_hat_axis, z_axis)
#        if np.linalg.norm(x_axis) < 1e-5:
#            if i > 0:
#                quaternions[i] = quaternions[i-1]
#            else:
#                quaternions[i] = np.array([0, 0, 0, 1])
#            continue
#        x_axis /= np.linalg.norm(x_axis)
#
#        y_axis = np.cross(z_axis, x_axis)
#
#        R_hand = np.column_stack([x_axis, y_axis, z_axis])
#        quaternions[i] = R.from_matrix(R_hand).as_quat()
#
#    return quaternions

def calculate_hand_orientation(pos_clean, joint_names, eps=1e-8):
    wrist_idx = joint_names.index("Right_Wrist")

    p_index_data = pos_clean[:, joint_names.index("Right_Index_Tip"), :]
    p_thumb_data = pos_clean[:, joint_names.index("Right_Thumb_Tip"), :]

    T = pos_clean.shape[0]
    q_xyzw = np.zeros((T, 4), dtype=float)

    p_w = pos_clean[:, wrist_idx, :]

    for k in range(T):
        c   = 0.5 * (p_thumb_data[k] + p_index_data[k])
        z   = c - p_w[k]

        nz  = np.linalg.norm(z)
        if nz < eps:
            q_xyzw[k] = q_xyzw[k-1] if k > 0 else np.array([0,0,0,1.0])
            continue
        z  /= nz

        u   = p_thumb_data[k] - p_index_data[k]

        # project u onto the plane orthogonal to z
        u  -= np.dot(u, z) * z
        nu  = np.linalg.norm(u)
        if nu < eps:
            q_xyzw[k] = q_xyzw[k-1] if k > 0 else np.array([0,0,0,1.0])
            continue
        y   = u / nu # Hand X (Red)

        x   = np.cross(z, y)                         # Hand Y (Green)
        # Orthonormalize again (Gram-Schmidt safety)
        x  -= np.dot(x, y) * y
        x  /= np.linalg.norm(x) + eps
        y   = np.cross(z, x)

        R_hand = np.column_stack([x, y, z])
        # Fix accidental left-handedness
        if np.linalg.det(R_hand) < 0:
            x = -x
            R_hand = np.column_stack([x, y, z])

        q_xyzw[k] = R.from_matrix(R_hand).as_quat()
    return q_xyzw
def smooth_orientation_trajectory(pos_clean, joint_names, window_length=15, polyorder=3):
    q_raw_xyzw = calculate_hand_orientation(pos_clean, joint_names)
    q_raw_wxyz = np.roll(q_raw_xyzw, 1, axis=-1)

    q_continuous_wxyz = enforce_hemisphere(q_raw_wxyz)

    e_raw = quaternion_log(q_continuous_wxyz)

    e_smoothed = np.empty_like(e_raw)
    for d in range(3):
        wl = min(window_length, len(e_raw) if len(e_raw) % 2 == 1 else len(e_raw)-1)
        wl = max(wl, 5)
        e_smoothed[:, d] = savgol_filter(e_raw[:, d], window_length=wl, polyorder=polyorder, mode="nearest")

    q_smoothed_wxyz = q_exp(e_smoothed)

    return q_normalize(q_smoothed_wxyz)