import numpy as np

names_ov1 = ["tau_ov1_split0_regr0_3d0_ov1_filters16", "tau_ov1_split0_regr0_3d0_prova1_ov1_32", "tau_ov1_split0_regr0_3d0_crnn_ov1_default_params"]
names_ov2 = ["tau_ov2_split0_regr0_3d0_ov2_filters16", "tau_ov2_split0_regr0_3d0_1000_epochs", "tau_ov2_split0_regr0_3d0_prova"]

data_ov1_x = []
data_ov1_y = []
for name in names_ov1:
  x_errors = np.load(name+"_x.npy")
  y_errors = np.load(name+"_y.npy")
  data_ov1_x.append(x_errors)
  data_ov1_y.append(y_errors)

data_ov1 = data_ov1_x+data_ov1_y

data_ov2_x = []
data_ov2_y = []
for name in names_ov2:
  x_errors = np.load(name+"_x.npy")
  y_errors = np.load(name+"_y.npy")
  data_ov2_x.append(x_errors)
  data_ov2_y.append(y_errors)

data_ov2 = data_ov2_x +data_ov2_y

labels = ["QCRNN 16 x", "QCRNN 32 x", "Seld-Net x", "QCRNN 16 y", "QCRNN 32 y", "Seld-Net y"]

import matplotlib.pyplot as plt
fig1, ax1 = plt.subplots()
ax1.set_title('confidence intervals on ov1')
ax1.set_xticklabels(labels,
                    rotation=25, fontsize=8)
ax1.boxplot(data_ov1,0, sym='')
fig1.savefig("ov1_conf")

fig2, ax2 = plt.subplots()
ax2.set_title('confidence intervals on ov2')
ax2.set_xticklabels(labels,
                    rotation=25, fontsize=8)
ax2.boxplot(data_ov2, 0, '')
fig2.savefig("ov2_conf")

