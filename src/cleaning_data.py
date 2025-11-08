import json
import numpy as np
from scipy.interpolate import interp1d

def load_demo_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    joint_names = [k for k in data[0].keys() if k not in ("frame","timestamp")]
    J = len(joint_names)
    T = len(data)
    pos = np.zeros((T, J, 3), dtype=np.float32)
    t = np.zeros(T, dtype=np.float64)
    for i, fr in enumerate(data):
        t[i] = float(fr["timestamp"])
        for j, name in enumerate(joint_names):
            coords = fr.get(name, [None, None, None])
            if any (c is None for c in coords):
                pos[i, j] = [np.nan, np.nan, np.nan]
            else:
                pos[i, j] = coords
    return t, pos, joint_names

def interpolate_nan_values(t, pos):
    T, J, D = pos.shape
    pos_inter = np.copy(pos)
    
    for j in range(J):
        for d in range(D):
            series = pos[:, j, d]
            valid_indices = np.where(np.isfinite(series))[0]
            
            if len(valid_indices) < 2:
                temp_series = pos_inter
                last_valid = None
                
                for i in range(T):
                    if np.isfinite(temp_series[i]):
                        last_valid = temp_series[i]
                    elif last_valid is not None:
                        temp_series[i] = last_valid
                first_valid = None
                for i in range(T-1, -1, -1):
                    if np.isfinite(temp_series[i]):
                        first_valid = temp_series[i]
                    elif first_valid is not None:
                        temp_series[i] = first_valid
                pos_inter[:, j, d] = temp_series
                continue
            interp_func = interp1d(t[valid_indices], series[valid_indices], kind='linear', bounds_error=False, fill_value=(series[valid_indices[0]], series[valid_indices[-1]]))
            
            nan_indices = np.where(np.isnan(series))[0]
            
            if len(nan_indices) > 0:
                pos_inter[nan_indices, j, d] = interp_func(t[nan_indices])
    if np.isnan(pos_inter).any():
        raise ValueError("Interpolation failed, NaN values remain.")
    return pos_inter

def ensure_monotonic_time(t, pos):
    order = np.argsort(t)
    return t[order], pos[order]

def madian_absolute_deviation(data, axis=None):
    med = np.nanmedian(data, axis=axis)
    return np.nanmedian(np.abs(data - med), axis=axis)

def mark_outlier_componentwise(pos, joint_names, k=4.0):
    pos2 = pos.copy()
    T, J, D = pos2.shape
    for j in range(J):
        for d in range(D):
            s = pos2[:, j, d]
            med = np.nanmedian(s)
            mad = madian_absolute_deviation(s)
            
            if not np.isfinite(mad) or mad == 0:
                q1, q3 = np.nanpercentile(s, [25, 75])
                iqr = q3 - q1 if q3 > q1 else 0.0
                
                lo = q1 - 3.0 * iqr
                hi = q3 + 3.0 * iqr
                
            else:
                lo = med - k * 1.4826 * mad
                hi = med + k * 1.4826 * mad
            mask = (s < lo) | (s > hi)
            pos2[mask, j, d] = np.nan
    return pos2

def mark_velocity_spikes(t, pos, vmax_per_joint=None, k=3.0):
    T, J, _ = pos.shape
    pos2 = pos.copy()
    dt = np.diff(t)
    dt[dt <= 0] = np.nan
    for j in range(J):
        p = pos2[:, j, :]
        dp = np.linalg.norm(np.diff(p, axis=0), axis=1)
        v = dp / dt
        if vmax_per_joint is None:
            med = np.nanmedian(v)
            mad = madian_absolute_deviation(v)
            if not np.isfinite(mad) or mad == 0:
                q1, q3 = np.nanpercentile(v, [25, 75])
                thr = (q3 + 3.0 * (q3 - q1)) if np.isfinite(q3) else np.inf
            else:
                thr = med + k * 1.4826 * mad
        else:
            thr = vmax_per_joint[j]
        spike_idx = np.where(v > thr)[0] 
        for i in spike_idx:
            pos2[i,   j, :] = np.nan
            pos2[i+1, j, :] = np.nan
    return pos2
        
def estimate_distance_bounds(pos, joint_a, joint_b, joint_names, k=4.0):
    
    ia, ib = joint_names.index(joint_a), joint_names.index(joint_b)
    da = pos[:, ia, :]
    db = pos[:, ib, :]
    
    d = np.linalg.norm(da - db, axis=1)
    med = np.nanmedian(d)
    mad = madian_absolute_deviation(d)
    
    if not np.isfinite(mad) or mad == 0:
        q1, q3 = np.nanpercentile(d, [25, 75])
        lo = max(0.0, q1 - 3.0 * (q3 - q1))
        hi = q3 + 3.0 * (q3 - q1)
    else:
        lo = max(0.0, med - k * 1.4826 * mad)
        hi = med + k * 1.4826 * mad
        
    return lo, hi

def mark_distance_violations(pos, joint_pairs, joint_names, bounds=None, k=4.0):
    pos2 = pos.copy()
    for (ja, jb) in joint_pairs:
        ia, ib = joint_names.index(ja), joint_names.index(jb)
        if bounds and (ja, jb) in bounds:
            lo, hi = bounds[(ja, jb)]
        else:
            lo, hi = estimate_distance_bounds(pos2, ja, jb, joint_names, k=k)
        da = pos2[:, ia, :]
        db = pos2[:, ib, :]
        d = np.linalg.norm(da - db, axis=1)
        
        mask = (d < lo) | (d > hi)
        med_a = np.nanmedian(pos2[:, ia, :], axis=0)
        med_b = np.nanmedian(pos2[:, ib, :], axis=0)
        dev_a = np.linalg.norm(da - med_a, axis=1)
        dev_b = np.linalg.norm(db - med_b, axis=1)
        
        pick_a = dev_a >= dev_b
        idx_a = np.where(mask & pick_a)[0]
        idx_b = np.where(mask & (~pick_a))[0]
        pos2[idx_a, ia, :] = np.nan

    return pos2

def clean_positions_with_constraints(t, pos, joint_names, component_k=4.0, 
                                     velocity_k=6.0, distance_k=4.0,
                                     joint_pairs=("Right_Wrist","Right_Thumb_Tip","Right_Index_Tip")):
    # 1) sort
    t, pos = ensure_monotonic_time(t, pos)

    # 2) initial fill for holes
    pos_clean = interpolate_nan_values(t, pos)

    # 3) component-wise robust bounds
    pos_clean = mark_outlier_componentwise(pos_clean, joint_names, k=component_k)
    pos_clean = interpolate_nan_values(t, pos_clean)

    # 4) velocity spikes
    pos_clean = mark_velocity_spikes(t, pos_clean, vmax_per_joint=None, k=velocity_k)
    pos_clean = interpolate_nan_values(t, pos_clean)

    # 5) distance constraints
    pairs = []
    if "Right_Wrist" in joint_names and "Right_Thumb_Tip" in joint_names:
        pairs.append(("Right_Wrist","Right_Thumb_Tip"))
    if "Right_Wrist" in joint_names and "Right_Index_Tip" in joint_names:
        pairs.append(("Right_Wrist","Right_Index_Tip"))
    if "Right_Shoulder" in joint_names and "Right_Elbow" in joint_names:                            # most likely not needed
        pairs.append(("Right_Shoulder","Right_Elbow"))                          # most likely not needed
    if "Right_Elbow" in joint_names and "Right_Wrist" in joint_names:                           # most likely not needed
        pairs.append(("Right_Elbow","Right_Wrist"))                         # most likely not needed

    if pairs:
        pos_clean = mark_distance_violations(pos_clean, pairs, joint_names, bounds=None, k=distance_k)
        pos_clean = interpolate_nan_values(t, pos_clean)

    return t, pos_clean