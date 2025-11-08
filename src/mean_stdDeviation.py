import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Mean Error Distance to the Goal
data = {
    "Goal Change": {
        "GMM": [0.00610, 0.00805, 0.00704, 0.00471, 0.01144, 0.00739, 0.00718, 0.00642, 0.01674, 0.00881],
        "LWR": [0.00521, 0.01997, 0.00549, 0.00548, 0.00506, 0.00538, 0.01217, 0.00510, 0.06678, 0.00542],
        "LR": [0.00290, 0.00297, 0.00290, 0.00795, 0.01333, 0.00284, 0.00290, 0.00272, 0.04507, 0.00120]
    },
    "Start Change": {
        "GMM": [0.00552, 0.01507, 0.00704, 0.00958, 0.00568, 0.00642, 0.00610, 0.00739, 0.00805, 0.00718],
        "LWR": [0.00738, 0.05879, 0.00549, 0.00571, 0.00463, 0.00510, 0.00521, 0.00538, 0.01997, 0.01217],
        "LR": [0.00493, 0.03967, 0.00322, 0.00290, 0.00290, 0.00290, 0.00297, 0.00284, 0.00795, 0.01333]
    },
    "Start and Goal Change": {
        "GMM": [0.00672, 0.00304, 0.00888, 0.00496, 0.01260, 0.00534, 0.01267, 0.01285, 0.00809, 0.00568],
        "LWR": [0.00529, 0.00438, 0.00512, 0.01199, 0.02767, 0.01208, 0.00637, 0.00655, 0.00618, 0.00463],
        "LR": [0.00363, 0.00363, 0.00795, 0.01859, 0.02397, 0.00290, 0.03969, 0.04509, 0.00789, 0.00290]
    }
}


print("--- Model Error Statistics ---")
print(f"{'Category':<25} | {'Model':<7} | {'Mean Error':<12} | {'Std. Deviation':<15}")
print("-" * 64)

summary_stats = []

for category, models in data.items():
    for model_name, errors in models.items():
        # Calculate mean and standard deviation
        mean_error = np.mean(errors)
        std_dev = np.std(errors)

        print(f"{category:<25} | {model_name:<7} | {mean_error:<12.4f} | {std_dev:<15.4f}")

        summary_stats.append({
            "Category": category,
            "Model": model_name,
            "Mean Error": mean_error,
            "Standard Deviation": std_dev
        })


plot_data = []
for category, models in data.items():
    for model_name, errors in models.items():
        for error_value in errors:
            plot_data.append({
                "Category": category,
                "Model": model_name,
                "Error": error_value
            })

df = pd.DataFrame(plot_data)

plt.figure(figsize=(12, 7))
sns.boxplot(data=df, x="Category", y="Error", hue="Model", palette="Set2")

plt.legend(title="Model", loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()