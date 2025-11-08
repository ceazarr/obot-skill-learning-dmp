import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math, os
from orientation_DMP import q_normalize, quaternion_multiply, quaternion_conjugate

def canonical_phase(t, alpha_x=math.log(100.0)):
    Ttot = max(t[-1] - t[0], 1e-12)
    tau  = (t - t[0]) / Ttot
    x = np.exp(-alpha_x * tau)
    return x, Ttot

def quat_geodesic_angle(q_ref_wxyz, q_est_wxyz):

    q1 = q_ref_wxyz / np.linalg.norm(q_ref_wxyz, axis=1, keepdims=True)
    q2 = q_est_wxyz / np.linalg.norm(q_est_wxyz, axis=1, keepdims=True)
    dots = np.abs(np.sum(q1*q2, axis=1))
    dots = np.clip(dots, 0.0, 1.0)
    return 2.0 * np.arccos(dots)  # radians

def quat_to_rpy(q_wxyz):

    q_xyzw = np.c_[q_wxyz[:,1], q_wxyz[:,2], q_wxyz[:,3], q_wxyz[:,0]]
    r = R.from_quat(q_xyzw)
    return r.as_euler('xyz', degrees=False)

def set_common_ylim(axes, series_list, pad=0.1):
    vmin = min(np.nanmin(s) for s in series_list)
    vmax = max(np.nanmax(s) for s in series_list)
    rng = vmax - vmin
    lo  = vmin - pad*rng
    hi  = vmax + pad*rng
    for ax in axes:
        ax.set_ylim(lo, hi)
def plot_orientation_overview(t, q_ref_wxyz, q_gen_wxyz, alpha_x=math.log(100.0), title="Orientation overview"):
    n = min(len(t), len(q_ref_wxyz), len(q_gen_wxyz))
    t = t[:n]; q_ref_wxyz = q_ref_wxyz[:n]; q_gen_wxyz = q_gen_wxyz[:n]

    rpy_ref = quat_to_rpy(q_ref_wxyz)     # (T,3)
    rpy_gen = quat_to_rpy(q_gen_wxyz)
    ang_err = quat_geodesic_angle(q_ref_wxyz, q_gen_wxyz)  # radians
    x, _    = canonical_phase(t, alpha_x=alpha_x)

    fig = plt.figure(figsize=(16, 12))
    labs = ["roll (rad)", "pitch (rad)", "yaw (rad)"]
    axes_rpy = []
    for i in range(3):
        ax = fig.add_subplot(3, 3, i+1)
        ax.plot(t, rpy_ref[:,i], label="ref", alpha=0.9)
        ax.plot(t, rpy_gen[:,i], "--", label="DMP", alpha=0.9)
        ax.set_xlabel("time (s)"); ax.set_ylabel(labs[i]); ax.grid(True)
        if i == 0: ax.legend()
        axes_rpy.append(ax)
    set_common_ylim(axes_rpy, [rpy_ref[:,0], rpy_ref[:,1], rpy_ref[:,2],
                               rpy_gen[:,0], rpy_gen[:,1], rpy_gen[:,2]], pad=0.1)

    # quaternion components
    axq = fig.add_subplot(3, 3, 4)
    axq.plot(t, q_ref_wxyz[:,0], label="w (ref)", alpha=0.9)
    axq.plot(t, q_gen_wxyz[:,0], "--", label="w (DMP)", alpha=0.9)
    axq.set_title("Quaternion w"); axq.set_xlabel("time (s)"); axq.grid(True)

    axqx = fig.add_subplot(3, 3, 5)
    axqx.plot(t, q_ref_wxyz[:,1], label="x (ref)", alpha=0.9)
    axqx.plot(t, q_gen_wxyz[:,1], "--", label="x (DMP)", alpha=0.9)
    axqx.set_title("Quaternion x"); axqx.set_xlabel("time (s)"); axqx.grid(True)

    axqy = fig.add_subplot(3, 3, 6)
    axqy.plot(t, q_ref_wxyz[:,2], label="y (ref)", alpha=0.9)
    axqy.plot(t, q_gen_wxyz[:,2], "--", label="y (DMP)", alpha=0.9)
    axqy.set_title("Quaternion y"); axqy.set_xlabel("time (s)"); axqy.grid(True)

    axqz = fig.add_subplot(3, 3, 7)
    axqz.plot(t, q_ref_wxyz[:,3], label="z (ref)", alpha=0.9)
    axqz.plot(t, q_gen_wxyz[:,3], "--", label="z (DMP)", alpha=0.9)
    axqz.set_title("Quaternion z"); axqz.set_xlabel("time (s)"); axqz.grid(True)
    axq.legend(loc="best")

    # phase + angular error
    axp = fig.add_subplot(3, 3, 8)
    axp.plot(t, x, label="phase x(t)")
    axp.set_title("Canonical phase"); axp.set_xlabel("time (s)"); axp.grid(True)

    axe = fig.add_subplot(3, 3, 9)
    axe.plot(t, ang_err, label="geodesic angle error")
    axe.set_title("Orientation error (rad)"); axe.set_xlabel("time (s)"); axe.grid(True)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig
def overlay_orientation_demos(demos, mode="phase", alpha_x=math.log(100.0)):

    lengths = np.array([len(d["t"]) for d in demos])
    ref = demos[np.argsort(lengths)[len(demos)//2]]

    t_ref = ref["t"]
    q_ref = ref["quat_wxyz"]
    rpy_ref = quat_to_rpy(q_ref)

    fig = plt.figure(figsize=(16, 10))
    axX = fig.add_subplot(2, 3, 1); axX.set_title("roll")
    axY = fig.add_subplot(2, 3, 2); axY.set_title("pitch")
    axZ = fig.add_subplot(2, 3, 3); axZ.set_title("yaw")
    axE = fig.add_subplot(2, 3, 4); axE.set_title("geodesic error to ref")
    axP = fig.add_subplot(2, 3, 5); axP.set_title("phase x(t)")

    r_all = []; p_all = []; y_all = []

    for d in demos:
        t = d["t"]; q = d["quat_wxyz"]
        if mode == "time":
            x_axis = t - t[0]
            rpy = quat_to_rpy(q)
        elif mode == "phase":
            x, _ = canonical_phase(t, alpha_x=alpha_x)
            x_axis = x
            rpy = quat_to_rpy(q)
        elif mode == "dtw":
            from fastdtw import fastdtw
            from scipy.spatial.distance import euclidean
            rpy = quat_to_rpy(q)
            dist, path = fastdtw(rpy, rpy_ref, dist=euclidean)

            buckets = [[] for _ in range(len(rpy_ref))]
            for i_src, i_ref in path:
                if 0 <= i_ref < len(rpy_ref):
                    buckets[i_ref].append(i_src)
            idx = np.array([int(np.median(b)) if b else 0 for b in buckets])
            idx = np.clip(idx, 0, len(rpy)-1)
            rpy = rpy[idx]
            x_axis = np.arange(len(rpy))
        else:
            raise ValueError("mode must be time|phase|dtw")

        axX.plot(x_axis, rpy[:,0], alpha=0.9, label=os.path.splitext(d["file"])[0])
        axY.plot(x_axis, rpy[:,1], alpha=0.9)
        axZ.plot(x_axis, rpy[:,2], alpha=0.9)
        r_all.append(rpy[:,0]); p_all.append(rpy[:,1]); y_all.append(rpy[:,2])

        if mode == "dtw":
            q_aligned = q[idx]
            t_aligned = np.linspace(t[0], t[-1], len(idx))
            q_ref_same = q_ref
            err = quat_geodesic_angle(q_ref_same, q_aligned)
            axE.plot(np.arange(len(err)), err, alpha=0.7)
        else:
            m = min(len(q), len(q_ref))
            err = quat_geodesic_angle(q_ref[:m], q[:m])
            axE.plot(np.arange(m), err, alpha=0.7)

        if mode in ("time","phase"):
            if mode == "time":
                axP.plot(x_axis, x_axis*0 + np.nan, alpha=0)
            else:
                axP.plot(x_axis, x_axis, alpha=0.4)

    set_common_ylim([axX], r_all, pad=0.1)
    set_common_ylim([axY], p_all, pad=0.1)
    set_common_ylim([axZ], y_all, pad=0.1)

    axX.set_xlabel(mode); axY.set_xlabel(mode); axZ.set_xlabel(mode)
    axE.set_xlabel("index"); axE.set_ylabel("rad")
    if mode == "phase": axP.set_xlabel("time"); axP.set_ylabel("x(t)")

    axX.legend(fontsize=8, ncols=2, loc="best")
    fig.tight_layout()
    return fig
def plot_axis_trajectory_on_sphere(q_wxyz, axis_local=np.array([0,0,1.0]), title="Axis trajectory"):
    q_xyzw = np.c_[q_wxyz[:,1], q_wxyz[:,2], q_wxyz[:,3], q_wxyz[:,0]]
    r = R.from_quat(q_xyzw)
    v = r.apply(axis_local)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(v[:,0], v[:,1], v[:,2], linewidth=2)
    ax.scatter(v[0,0], v[0,1], v[0,2], s=60, label="start")
    ax.scatter(v[-1,0],v[-1,1],v[-1,2],s=60,label="end")
    # draw unit sphere wireframe (light)
    u = np.linspace(0, 2*np.pi, 40)
    vth = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(vth))
    ys = np.outer(np.sin(u), np.sin(vth))
    zs = np.outer(np.ones_like(u), np.cos(vth))
    ax.plot_wireframe(xs, ys, zs, linewidth=0.3, alpha=0.3)
    ax.set_title(title); ax.legend()
    ax.set_box_aspect([1,1,1])
    return fig

def quat_rotate(q, v):
    qv = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
    return quaternion_multiply(quaternion_multiply(q, qv), quaternion_conjugate(q))[..., 1:]

def quat_to_axis(q, axis='z'):
    base = {'x': np.array([1,0,0.0]),
            'y': np.array([0,1,0.0]),
            'z': np.array([0,0,1.0])}[axis]
    base = np.broadcast_to(base, q.shape[:-1] + (3,))
    return quat_rotate(q, base)

def angular_error_deg(q_ref, q_est):

    dots = np.abs(np.sum(q_ref * q_est, axis=-1))
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))
def plot_axis_trajectories_on_sphere_overlay(
        q_ref_wxyz, q_dmp_wxyz, axis='z', title="Axis trajectory on unit sphere",
        show=True, block=None
):

    q_ref = q_normalize(q_ref_wxyz)
    q_dmp = q_normalize(q_dmp_wxyz)

    Tref, Tdmp = len(q_ref), len(q_dmp)
    if Tref != Tdmp:
        if Tref > Tdmp:
            idx = np.linspace(0, Tref-1, Tdmp).astype(int)
            q_ref = q_ref[idx]
        else:
            idx = np.linspace(0, Tdmp-1, Tref).astype(int)
            q_dmp = q_dmp[idx]

    v_ref = quat_to_axis(q_ref, axis=axis)
    v_dmp = quat_to_axis(q_dmp, axis=axis)

    ang_err = angular_error_deg(q_ref, q_dmp)
    print(f"[sphere] {axis}-axis angular error — mean: {ang_err.mean():.2f}°, max: {ang_err.max():.2f}°")

    # unit sphere mesh
    u = np.linspace(0, 2*np.pi, 120)
    v = np.linspace(0, np.pi, 60)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xs, ys, zs, alpha=0.1, linewidth=0)

    for th in np.linspace(0, np.pi, 7):
        ax.plot(np.cos(u)*np.sin(th), np.sin(u)*np.sin(th), np.cos(th)*np.ones_like(u), alpha=0.15, linewidth=0.7)
    for ph in np.linspace(0, 2*np.pi, 12, endpoint=False):
        ax.plot(np.cos(ph)*np.sin(v), np.sin(ph)*np.sin(v), np.cos(v), alpha=0.15, linewidth=0.7)

    # trajectories
    ax.plot(v_ref[:,0], v_ref[:,1], v_ref[:,2], label="Ref", linewidth=2)
    ax.plot(v_dmp[:,0], v_dmp[:,1], v_dmp[:,2], label="DMP", linewidth=2, linestyle='--')

    ax.scatter(v_ref[0,0], v_ref[0,1], v_ref[0,2], s=60, marker='o', label="Ref start")
    ax.scatter(v_ref[-1,0], v_ref[-1,1], v_ref[-1,2], s=120, marker='*', label="Ref end")
    ax.scatter(v_dmp[0,0], v_dmp[0,1], v_dmp[0,2], s=60, marker='o', label="DMP start")
    ax.scatter(v_dmp[-1,0], v_dmp[-1,1], v_dmp[-1,2], s=120, marker='*', label="DMP end")

    ax.set_title(f"{title}\n(axis = {axis})")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_box_aspect([1,1,1])
    ax.legend(loc="upper left", fontsize=9)
    lim = 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)

    if show:
        plt.show(block=block)
    return fig