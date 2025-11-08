from scipy.signal import savgol_filter
import numpy as np

def q_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return q / n

def quaternion_multiply(q1, q2):
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    # broadcast to common shape
    q1, q2 = np.broadcast_arrays(q1, q2)
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.moveaxis(np.stack([w, x, y, z], axis=0), 0, -1)

def quaternion_log(q):
    q = q_normalize(q)
    w = q[..., :1]
    v = q[..., 1:]
    nv = np.linalg.norm(v, axis=-1, keepdims=True)
    small = nv < 1e-8
    w_clip = np.clip(w, -1.0, 1.0)
    theta = np.arccos(w_clip)
    # e = theta * v / ||v|| ; for small use first-order
    scale = np.where(small, 1.0, theta / np.where(small, 1.0, nv))
    return scale * v

def q_exp(r):
    r = np.asarray(r, dtype=float)
    if r.ndim == 1:
        r = r.reshape(1, 3)
    theta = np.linalg.norm(r, axis=-1, keepdims=True)
    small = theta < 1e-8
    w = np.where(small, 1.0, np.cos(theta))
    u = np.where(small, 0.5 * r, (np.sin(theta) / theta) * r)
    return q_normalize(np.concatenate([w, u], axis=-1))

def enforce_hemisphere(qs):
    qs = np.asarray(qs, dtype=float).copy()
    for i in range(1, len(qs)):
        if np.dot(qs[i-1], qs[i]) < 0.0:
            qs[i] = -qs[i]
    return qs


def quaternion_conjugate(q):
    q = np.asarray(q, dtype=float)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out

def derivatives_quat(E, t, window_length=11, polyorder=3):
    T = len(t)
    w1 = min(window_length, T if T % 2 == 1 else T - 1)
    w1 = max(w1, 3)
    dt_mean = np.mean(np.diff(t)) if T > 1 else 1.0
    E0 = np.empty_like(E); E1 = np.empty_like(E); E2 = np.empty_like(E)
    for d in range(3):
        try:
            E0[:, d] = savgol_filter(E[:, d], window_length=w1, polyorder=polyorder, mode="nearest")
            E1[:, d] = savgol_filter(E[:, d], window_length=w1, polyorder=polyorder, deriv=1, delta=dt_mean, mode="nearest")
            E2[:, d] = savgol_filter(E[:, d], window_length=w1, polyorder=polyorder, deriv=2, delta=dt_mean, mode="nearest")
        except Exception:
            from scipy import ndimage
            sigma = 1.5
            E0[:, d] = ndimage.gaussian_filter1d(E[:, d], sigma=sigma, mode="nearest")
            E1[:, d] = ndimage.gaussian_filter1d(E[:, d], sigma=sigma, order=1, mode="nearest") / dt_mean
            E2[:, d] = ndimage.gaussian_filter1d(E[:, d], sigma=sigma, order=2, mode="nearest") / (dt_mean ** 2)
    return E0, E1, E2