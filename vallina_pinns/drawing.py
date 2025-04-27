import numpy as np
import matplotlib.pyplot as plt

def plot2d(X_star,u_star,pred):
    fig = plt.figure(figsize=[10,3])
    axes = fig.subplots(1,3, sharex=False, sharey=False)
    img1 = axes[0].scatter(X_star[:, 0:1], X_star[:, 1:2], c=u_star, cmap='jet', vmax=1, vmin=-1, s=5)
    axes[0].set_title('Reference solution', fontsize=15)
    axes[0].set_xlabel('x', fontsize=15)
    axes[0].set_ylabel('t', fontsize=15)
    plt.colorbar(img1, ax=axes[0])
    img2 = axes[1].scatter(X_star[:, 0:1], X_star[:, 1:2], c=pred, cmap='jet', vmax=1, vmin=-1, s=5)
    axes[1].set_title('PINNs prediction', fontsize=15)
    axes[1].set_xlabel('x', fontsize=15)
    axes[1].set_ylabel('t', fontsize=15)
    plt.colorbar(img2, ax=axes[1])
    img3 = axes[2].scatter(X_star[:, 0:1], X_star[:, 1:2], c=u_star-pred, cmap='seismic', vmax=0.01, vmin=-0.01, s=5)
    axes[2].set_title('Error', fontsize=15)
    axes[2].set_xlabel('x', fontsize=15)
    axes[2].set_ylabel('t', fontsize=15)
    plt.colorbar(img3, ax=axes[2])
    plt.show()