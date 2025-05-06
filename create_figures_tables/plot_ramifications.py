import os
import matplotlib.pyplot as plt
import numpy as np

from plot_config import mpl

# Define the save path
save_path = "."

# HDC position encoding
b = np.pi/10;
t = np.arange(10)
p = np.cos(b*t)

# Define the binary feature sets
features = {
    'A': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'B': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'C': [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
    'D': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
}

# Plot and save each signal separately
for key, values in features.items():
    plt.figure(figsize=(10, 6))
    markerline, stemlines, baseline = plt.stem(values, linefmt='-', markerfmt='o', basefmt=' ')

    # Style adjustments
    plt.setp(markerline, color='orange', markersize=10, markeredgewidth=2)
    plt.setp(stemlines, color='blue', linewidth=2)
    plt.setp(baseline, color='black', linewidth=1)

    plt.title(f'Signal {key}', fontsize=30)
    plt.ylim(-0.5, max(values) + 0.5)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Time', fontsize=25)
    plt.ylabel('Amplitude', fontsize=25)

    # Save the figure
    plt.savefig(os.path.join(f"{save_path}/images", f'{key}_signal.png'))
    plt.close()

# print ppv-hdc for all signals
for key, values in features.items():
    ppv = np.sum(values * p)
    print(f"PPV for signal {key}: {ppv:.2f}")

# Define the binary signals
signals = {
    'A': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'B': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'C': [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    'D': [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'E': [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
    'F': [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    'G': [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    'H': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    'I': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    'H': [0, 0, 0, 0, 0, 0, 10, 10, 10, 10]
}

# Calculate b, t, p
b = np.pi / 10
t = np.arange(10)
p = np.cos(b * t)

# Compute HDC-PPV for each signal A-H
for key, values in signals.items():
    ppv = np.sum(values * p)
    print(f"PPV for signal {key}: {ppv:.2f}")