import matplotlib.pyplot as plt
import numpy as np

# Generate random data for the boxplot
# feature_1 = np.random.normal(loc=0.0, scale=1.0, size=100)  # ndarray
# feature_2 = np.random.normal(loc=0.0, scale=2.0, size=100)
# feature_3 = np.random.normal(loc=0.0, scale=3.0, size=100)


level = np.array([0.0024, 0.0021, 0.0025   ])
fanout = np.array([0.002, 0.003, 0.004   ])
cut_fanout = np.array([0.003, 0.0041, 0.0053   ])
delay = np.array([0.005, 0.006, 0.002   ])
delay1 = np.array([0.0011, 0.0014, 0.0021   ])
delay2 = np.array([0.0021, 0.0011, 0.0017   ])
delay3 = np.array([0.0014, 0.0012, 0.0014   ])
delay4 = np.array([0.0002, 0.0001, 0.0002   ])
delay5 = np.array([0.0001, 0.0001, 0.0001 ])
leakpower = np.array([0.0001, 0.0001, 0.0002 ])
area = np.array([0.0001, 0.0002, 0.0004  ])
inverter = np.array([0.0002, 0.0001, 0.0001  ])
cut_leaf = np.array([0.002, 0.0014, 0.0013  ])
fanout_gap = np.array([0.0001, 0.0004, 0.0003  ])





data = [area, leakpower,  cut_leaf, fanout_gap,  level, fanout, inverter,  delay1, delay2,delay3,delay4,delay5,delay]

# Create a figure and axes
fig, ax = plt.subplots(figsize=(6, 8))
# Create the boxplot
# Customize the boxplot properties
boxprops = dict(color='blue', linewidth=2.0, edgecolor='black', facecolor='lightblue')
whiskerprops = dict(color='gray', linestyle='--', linewidth=1.0)
medianprops = dict(color='red', linewidth=2.0)
flierprops = dict(marker='o', markersize=5, markerfacecolor='green', markeredgecolor='black')
capprops = dict(color='black', linewidth=2.0)

# Create the boxplot
ax.boxplot(data, vert=False, patch_artist=True, boxprops=boxprops, whiskerprops=whiskerprops,
           medianprops=medianprops, flierprops=flierprops, capprops=capprops)


fontsize = 14

# Customize the plot
# ax.set_xlabel('Diff. Value', fontsize=fontsize)
# ax.set_ylabel('Features', fontsize=fontsize)


# x_ticks = np.linspace(0, 5, 10)
x_ticks = [0.000, 0.002, 0.004,0.006, 0.008 ]

ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks, fontsize=fontsize)


y_ticks = ['0', 'area', 'leakage power',  'cut leaves', 'fanout gap',  'level', 'node/cut fanout', 'inverter',  'pin1 delay', 'pin2 delay','pin3 delay','pin4 delay','pin5 delay','overll delay']
# y_ticks = [1,2,3,4,5,6,7, 9, 9, 10, 11, 12, 13]
# ax.set_yticks(y_ticks)
# ax.set_yticklabels(y_ticks, fontsize=fontsize)
plt.yticks(range(0,len(y_ticks)), y_ticks, fontsize=14)
plt.title("Permutation Importance")

# Show the plot
plt.tight_layout()
# plt.show()
plt.savefig('boxplot.pdf', dpi=800)

