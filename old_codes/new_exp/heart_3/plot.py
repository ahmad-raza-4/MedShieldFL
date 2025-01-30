# Plot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")

# Read and preprocess the data
results = pd.read_csv("results_fed_heart_disease.csv")
results = results.rename(columns={"perf": "Performance"})

# Define linestyles
linestyle_str = [
    ("solid", "solid"),        # Same as (0, ()) or '-'
    ("dotted", "dotted"),      # Same as (0, (1, 1)) or ':'
    ("dashed", "dashed"),      # Same as '--'
    ("dashdot", "dashdot"),
]
linestyle_tuple = [
    ("loosely dotted", (0, (1, 10))),
    ("densely dotted", (0, (1, 1))),
    ("loosely dashed", (0, (5, 10))),
    ("densely dashed", (0, (5, 1))),
    ("loosely dashdotted", (0, (3, 10, 1, 10))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
]
linestyles = linestyle_tuple + linestyle_str

# Extract unique delta values, excluding NaNs
deltas = [d for d in results["d"].unique() if not (pd.isna(d))]

fig, ax = plt.subplots(figsize=(10, 6))  # Optional: Specify figure size for better readability

# Plot lines for each delta
for i, d in enumerate(deltas):
    cdf = results.loc[results["d"] == d]
    sns.lineplot(
        data=cdf,
        x="e",
        y="Performance",
        label=f"delta={d}",
        linestyle=linestyles[::-1][i][1],
        ax=ax,
    )

# Set x-axis to logarithmic scale
ax.set_xscale("log")

# Updated tick values and labels
xtick_values = [1, 10, 20, 30]
xlabels = [str(v) for v in xtick_values]
ax.set_xticks(xtick_values)
ax.set_xticklabels(xlabels)

# Calculate and plot the baseline if available
baseline = results.loc[results["d"].isnull(), "Performance"]
if not baseline.empty:
    baseline_mean = baseline.mean()
    ax.axhline(
        baseline_mean,
        color="black",
        label="Baseline wo DP",
        linestyle="--",  # Optional: Differentiate baseline with a distinct linestyle
        linewidth=2,     # Optional: Make the baseline more prominent
    )
else:
    print("Warning: No baseline data available (no rows with d as null). Skipping baseline line.")

# Set x-axis limits
ax.set_xlim(1, 30)

# Final plot adjustments
plt.legend()
plt.xlabel("epsilon")
plt.ylabel("Performance")
plt.title("Performance vs Epsilon for Heart Disease Dataset with Differential Privacy")  # Optional: Add a title
plt.tight_layout()  # Adjust layout for better fit

# Save the figure
plt.savefig("perf_function_of_dp_heart_disease_updated.pdf", dpi=100, bbox_inches="tight")
plt.show()  # Optional: Display the plot interactively
