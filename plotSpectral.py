import matplotlib.pyplot as plt


acc = [94.38, 95.34, 94.79, 94.92, 94.84]
n_spectral = [2, 3, 4, 5, 6]

fig, ax = plt.subplots()
plt.plot(n_spectral, acc, color='green', lw=1, marker='*', ms=10, label='SNet')
plt.scatter(0, 94.15, color='red', label='UNet')
plt.ylabel('acc /%')
plt.xlabel('num of spectral')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend()
plt.show()