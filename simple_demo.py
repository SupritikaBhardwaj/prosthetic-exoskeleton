"""
Simple Real-Time Monitoring Demo
Neuro-Controlled Robotic Limb - Basic Visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import time

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle("Neuro-Controlled Robotic Limb - Real-Time Monitoring", fontsize=20)

# Labels for each panel
titles = ["EMG Muscle Activation", "IMU Motion (Accelerometer)", 
          "Predicted Joint Angle", "Balance Score"]

# Initialize empty plots
lines = []

for ax, title in zip(axs.flatten(), titles):
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.set_ylim(-2, 2)
    line, = ax.plot([], [])
    lines.append(line)

# Data buffers
data = [[], [], [], []]

# Real-time simulation
for t in range(1000):
    # Fake data for simulation
    data[0].append(np.sin(t*0.1))                # EMG signal
    data[1].append(np.cos(t*0.1))                # IMU signal
    data[2].append(np.sin(t*0.05) * 1.2)         # Joint angle prediction
    data[3].append(np.random.uniform(-1, 1))     # Balance score
    
    # Update graph
    for i in range(4):
        lines[i].set_data(range(len(data[i])), data[i])
    
    plt.pause(0.01)

plt.show()

