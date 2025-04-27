import numpy as np
import matplotlib.pyplot as plt
from math import *
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.keras.backend.set_floatx('float64')
tf.get_logger().setLevel('ERROR')


nx = 256
nt = 100
x = np.linspace(-1, 1, nx)
t = np.linspace(0, 1, nt)
X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

idx_init = np.where(X_star[:, 1]==0)[0]
X_init = X_star[idx_init]
u_init = -np.sin(pi*X_init[:, 0:1])

# Define the BC:
idx_bc = np.where((X_star[:, 0]==1.0)|(X_star[:, 0]==-1.0))[0]
X_bc = X_star[idx_bc]
u_bc = np.zeros((X_bc.shape[0], 1))

# Define training collocation points
N_f = 2000
idx_Xf = np.random.choice(X_star.shape[0], N_f, replace=False)#随机选2000个点
X_colloc_train = X_star[idx_Xf]

# Define the PDE parameter
nu = 0.025

@tf.function
def net_transform(X_f, model_nn):
    return model_nn(X_f)

@tf.function
def f_user(X_f, model_nn):
    x_temp = X_f[:, 0:1]
    t_temp = X_f[:, 1:2]
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_temp)
        tape.watch(t_temp)
        X_temp = tf.concat([x_temp, t_temp], axis=1)

        u = net_transform(X_temp, model_nn)

        u_x = tape.gradient(u, x_temp)
        u_xx = tape.gradient(u_x, x_temp)
        u_t = tape.gradient(u, t_temp)

    f = u_t + u*u_x - nu*u_xx
    del tape
    return f

@tf.function
def loss_f(f):
    return tf.reduce_mean(tf.square(f))

inputs = keras.Input(shape=(2,))
hidden_layers=[]
for _ in range(4):
    hidden_layers.append(layers.Dense(50, activation='tanh'))
x = hidden_layers[0](inputs)
for layer in hidden_layers[1:]:
    x = layer(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs=inputs, outputs=outputs, name="pinns")
model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=2e-4)
start_time=time.time()
for i in range(1000):
    with tf.GradientTape() as tape2:
        bc_pred=net_transform(X_bc,model)
        ic_pred=net_transform(X_init,model)
        pde_loss=f_user(X_colloc_train,model)
        loss=loss_f(bc_pred-u_bc)+loss_f(ic_pred-u_init)+loss_f(pde_loss)
    grads = tape2.gradient(loss, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))
    if i%100==0:
        print("step: ",i," total loss: ",loss.numpy())
end_time=start_time-time.time()
print("total time: ",end_time)
