import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from drawing import plot2d
pinn=keras.models.load_model('pinn')

data = scipy.io.loadmat('W:/documents/dev/python/PINNs/PINNs/vallina_pinns/burgers_shock.mat')
x = np.linspace(-1, 1, 256)
t = np.linspace(0, 1, 100)
X, T = np.meshgrid(x,t)
xt = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))  # (25600, 2)
x_=xt[:,0:1]
t_=xt[:,1:2]
u_exact = np.real(data['usol']).T  # (100, 256)

idx_init = np.where(xt[:, 1]==0)[0]
xt_init = xt[idx_init] # (256, 2)
x_init=xt_init[:,0:1]
t_init=xt_init[:,1:2]
u_init = -np.sin(np.pi*xt_init[:, 0:1])   # (256, 1)

idx_bc = np.where((xt[:, 0]==1.0)|(xt[:, 0]==-1.0))[0]
xt_bc = xt[idx_bc]  # (200, 2)
x_bc=xt_bc[:,0:1]
t_bc=xt_bc[:,1:2]
u_bc = np.zeros((xt_bc.shape[0], 1))  # (200, 1)

idx_Xf = np.random.choice(xt.shape[0], 2000, replace=False)#随机选2000个点
collocation_points = xt[idx_Xf] # (2000,2)
x_collocation_points=collocation_points[:,0:1]
t_collocation_points=collocation_points[:,1:2]

xt,u_exact,xt_init,u_init,xt_bc,u_bc,collocation_points,x_,t_=map(lambda x: tf.convert_to_tensor(x,dtype=tf.float64),[xt,u_exact,xt_init,u_init,xt_bc,u_bc,collocation_points,x_,t_])
u_pred=pinn(tf.concat([x_, t_], axis=1))
u_pred=u_pred.numpy()
u_pred=u_pred.reshape(100,256)

plot2d(xt,u_exact,u_pred)

