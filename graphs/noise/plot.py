import matplotlib.pyplot as plt
import numpy as np

# Data
epochs = [1, 2, 3, 4, 5]
base_nm_e1 = [4.88, 4.88, 4.88, 4.88, 4.88]
dynamic_nm_e1 = [0.88, 0.98, 1.07, 1.09, 1.13]
base_nm_e10 = [0.84, 0.84, 0.84, 0.84, 0.84]
dynamic_nm_e10 = [0.15, 0.18, 0.20, 0.20, 0.21]
base_nm_e30 = [0.41, 0.41, 0.41, 0.41, 0.41]
dynamic_nm_e30 = [0.07, 0.09, 0.10, 0.10, 0.10]

# Plotting
plt.figure(figsize=(10, 6))

# Plot for e=1.0
plt.plot(epochs, base_nm_e1, label='Base NM (e=1.0)', marker='o')
plt.plot(epochs, dynamic_nm_e1, label='Dynamic NM (e=1.0)', marker='o')

# Plot for e=10.0
plt.plot(epochs, base_nm_e10, label='Base NM (e=10.0)', marker='s')
plt.plot(epochs, dynamic_nm_e10, label='Dynamic NM (e=10.0)', marker='s')

# Plot for e=30.0
plt.plot(epochs, base_nm_e30, label='Base NM (e=30.0)', marker='^')
plt.plot(epochs, dynamic_nm_e30, label='Dynamic NM (e=30.0)', marker='^')

# Adding labels and title
plt.xlabel('Epoch')
plt.ylabel('NM Value')
plt.title('Base NM vs Dynamic NM across Epochs for Different Epsilon Values')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('nm_comparison_plot.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()