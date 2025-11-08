import json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

def save_orientation_model(path, model, meta=None):

    gmm = model["gmm"]
    out = {
        "alpha_z": float(model["alpha_z"]),
        "beta_z":  float(model["beta_z"]),
        "alpha_x": float(model["alpha_x"]),
        "tau":     float(model["tau"]),
        "gmm": {
            "covariance_type": gmm.covariance_type,
            "weights": gmm.weights_.tolist(),
            "means": gmm.means_.tolist(),
            "covariances": gmm.covariances_.tolist()
        },
        "meta": meta or {}
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[save] Orientation model -> {path}")

def load_orientation_model(path):

    with open(path, "r") as f:
        data = json.load(f)

    g = data["gmm"]
    n_components = len(g["weights"])
    gmm = GaussianMixture(n_components=n_components, covariance_type=g["covariance_type"])
    gmm.weights_     = np.array(g["weights"], dtype=float)
    gmm.means_       = np.array(g["means"], dtype=float)
    gmm.covariances_ = np.array(g["covariances"], dtype=float)
    gmm.precisions_cholesky_ = _compute_precision_cholesky(gmm.covariances_, gmm.covariance_type)

    model = {
        "gmm": gmm,
        "alpha_z": float(data["alpha_z"]),
        "beta_z":  float(data["beta_z"]),
        "alpha_x": float(data["alpha_x"]),
        "tau":     float(data["tau"]),
    }
    return model, data.get("meta", {})
